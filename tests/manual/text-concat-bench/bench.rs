// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Benchmark for text rendering with concatenated bindings on the embedded
//! software renderer path (bitmap fonts, no systemfonts), mirroring an MCU
//! setup: software renderer, `EmbedForSoftwareRenderer` fonts, minimal
//! window with swapped buffers, single-threaded, no event loop.
//!
//! Three phases, each reported as median over many frames:
//! - unchanged redraw: force a redraw without touching any property. The
//!   floor cost of a frame (dirty tracking skips everything).
//! - color update: change a color property so the text items redraw with
//!   unchanged strings — the redraw a layout cache can serve without
//!   re-shaping or re-breaking.
//! - counter update: increment the value the text bindings read. The string
//!   changes, so the layout must recompute; this is the per-update cost a
//!   UI pays for a changing value.
//!
//! Frames are deterministic, so rendering changes can be checked for pixel
//! regressions by dumping the first frames of each phase to PPM images and
//! diffing them across code versions:
//!
//! ```sh
//! ./text-concat-bench frames-before
//! # ... apply the change, rebuild, run again ...
//! ./text-concat-bench frames-after
//! cmp frames-before/initial.ppm frames-after/initial.ppm
//! ```

use slint::Model as _;
use slint::platform::software_renderer::{
    LineBufferProvider, MinimalSoftwareWindow, Rgb565Pixel,
};
/// Test-only renderer types: the binary measures through `main`, only the
/// tests compare dirty regions, rotations and buffer types.
#[cfg(test)]
use slint::platform::software_renderer::{PhysicalRegion, RenderingRotation, RepaintBufferType};
use slint::platform::{PlatformError, WindowAdapter};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

slint::include_modules!();

use core::sync::atomic::{AtomicU64, Ordering};
/// A counting allocator: allocation counts and net bytes are deterministic
/// on the single-threaded bench, which turns them into CI-safe regression
/// gates for per-frame work and for the memory the text caches retain.
use std::alloc::{GlobalAlloc, Layout, System};

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding to the system allocator with the caller's
        // layout, exactly the contract of a GlobalAlloc implementation.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout come from the paired alloc call.
        unsafe { System.dealloc(pointer, layout) };
        ALLOC_BYTES.fetch_sub(layout.size() as u64, Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Serializes the tests in this crate. The allocation counters are
/// process-global, so two tests measuring allocations concurrently would
/// contaminate each other's snapshots. Every test holds this guard for its
/// whole body; a poisoned mutex still yields its guard because a panicking
/// test must not unlock concurrent allocation measurements.
#[cfg(test)]
static TEST_SERIALIZER: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
fn serialized_test() -> std::sync::MutexGuard<'static, ()> {
    TEST_SERIALIZER.lock().unwrap_or_else(|poison| poison.into_inner())
}

thread_local! {
    static WINDOW: Rc<MinimalSoftwareWindow> = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::SwappedBuffers,
    );
}

struct BenchPlatform;
impl slint::platform::Platform for BenchPlatform {
    fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, PlatformError> {
        Ok(WINDOW.with(|x| x.clone()))
    }
}

const SIZE: slint::PhysicalSize = slint::PhysicalSize { width: 480, height: 1060 };
const WARMUP_FRAMES: u32 = 10;
const FRAMES: u32 = 400;

/// The whole bench allocates through [`CountingAllocator`]; a phase takes
/// snapshots of the counters to report what one frame allocates and how the
/// net retained bytes move.
#[derive(Clone, Copy)]
struct AllocSnapshot {
    calls: u64,
    bytes: u64,
}

impl AllocSnapshot {
    fn take() -> Self {
        Self {
            calls: ALLOC_CALLS.load(Ordering::Relaxed),
            bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        }
    }

    fn delta_since(&self, other: &Self) -> (u64, i64) {
        (self.calls - other.calls, self.bytes as i64 - other.bytes as i64)
    }
}

fn do_rendering(window: &MinimalSoftwareWindow, buffer: &mut [Rgb565Pixel]) -> bool {
    window.request_redraw();
    window.draw_if_needed(|renderer| {
        renderer.render(buffer, SIZE.width as usize);
    })
}

/// A line-buffer view over the full frame buffer, for measuring the
/// render-by-line path (the rasterization mode of smaller MCU panels).
struct FullBuffer<'a>(&'a mut [Rgb565Pixel]);

impl LineBufferProvider for FullBuffer<'_> {
    type TargetPixel = Rgb565Pixel;
    fn process_line(
        &mut self,
        line: usize,
        range: core::ops::Range<usize>,
        render_fn: impl FnOnce(&mut [Self::TargetPixel]),
    ) {
        render_fn(&mut self.0[line * SIZE.width as usize..][range]);
    }
}

/// Writes the frames of a phase to `{name}.ppm` files when a dump directory
/// is configured, for pixel-diffing across code versions.
struct FrameDump {
    dir: Option<PathBuf>,
}

impl FrameDump {
    fn dump(&self, name: &str, index: u32, buffer: &[Rgb565Pixel]) {
        let Some(dir) = &self.dir else { return };
        let _ = std::fs::create_dir_all(dir);
        let mut file = match std::fs::File::create(dir.join(format!("{name}-{index}.ppm"))) {
            Ok(file) => file,
            Err(err) => {
                eprintln!("skipping frame dump {name}-{index}: {err}");
                return;
            }
        };
        use std::io::Write;
        let _ = write!(file, "P6\n{} {}\n255\n", SIZE.width, SIZE.height);
        let mut data = Vec::with_capacity(buffer.len() * 3);
        for pixel in buffer {
            // Decode the RGB565 layout into the top-aligned 8-bit channels,
            // the same way the renderer's own pixel type does.
            data.push(((pixel.0 & 0b1111_1000_0000_0000) >> 8) as u8);
            data.push((((pixel.0 & 0b0000_0111_1110_0000) >> 3) as u8) << 2);
            data.push(((pixel.0 & 0b0000_0000_0001_1111) as u8) << 3);
        }
        let _ = file.write_all(&data);
    }
}

struct Harness {
    ui: ConcatBench,
    window: Rc<MinimalSoftwareWindow>,
    buffer: Vec<Rgb565Pixel>,
}

impl Harness {
    fn new() -> Self {
        let _ = slint::platform::set_platform(Box::new(BenchPlatform));
        let ui = ConcatBench::new().unwrap();
        seed_dynamic_content(&ui);
        let _ = ui.show();
        let window = WINDOW.with(|x| x.clone());
        window.set_size(SIZE);
        let buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];
        let mut harness = Harness { ui, window, buffer };
        // First rendering evaluates bindings and primes the double buffer.
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        harness
    }
}

/// Seeds the model-driven and long-token content used by both the benchmark
/// and the pixel tests. Pixel tests must call this too: mutating rows of an
/// empty model is a no-op, which would make the list-row assertions vacuous.
fn seed_dynamic_content(ui: &ConcatBench) {
    let rows = slint::VecModel::from(
        (0..8).map(|row| format!("{row} down").into()).collect::<Vec<_>>(),
    );
    ui.set_list_rows(slint::ModelRc::from(Rc::new(rows)));
    ui.set_long_token("a".repeat(240).into());
}

fn phase<F: FnMut(u32)>(name: &str, frames: u32, mut frame: F) {
    // Warm up caches so the measured frames run steady-state.
    for _ in 0..WARMUP_FRAMES {
        frame(u32::MAX);
    }
    let mut samples = Vec::new();
    let mut alloc_calls = 0u64;
    let mut alloc_net = 0i64;
    let mut samples_frames = 0u32;
    for index in 0..frames {
        let before = AllocSnapshot::take();
        let start = Instant::now();
        frame(index);
        let elapsed = start.elapsed();
        let after = AllocSnapshot::take();
        let (calls, net) = after.delta_since(&before);
        alloc_calls = alloc_calls.max(calls);
        alloc_net = alloc_net.max(net);
        samples.push(elapsed);
        samples_frames += 1;
    }
    samples.sort();
    let median = samples[samples.len() / 2].as_nanos() as f64;
    let p95 = samples[samples.len() * 95 / 100].as_nanos() as f64;
    println!(
        "{name:26} median {median:10.0} ns   p95 {p95:10.0} ns   allocs/frame <= {alloc_calls:4}   retained <= {alloc_net:+6} B"
    );
    let _ = samples_frames;
}

fn main() {
    let dump = FrameDump { dir: std::env::args().nth(1).map(PathBuf::from) };
    let mut harness = Harness::new();
    dump.dump("initial", 0, &harness.buffer);

    let mut toggle = false;
    let mut counter = 0;
    let mut late_counter = 0;
    let mut words = ["gamma", "delta", "theta", "alpha"].iter().cycle();

    phase("unchanged redraw", FRAMES, |index| {
        harness.window.request_redraw();
        let drew = harness.window.draw_if_needed(|renderer| {
            renderer.render(&mut harness.buffer, SIZE.width as usize);
        });
        if drew && index < 3 {
            dump.dump("unchanged-redraw", index, &harness.buffer);
        }
    });

    phase("color update", FRAMES, |index| {
        toggle = !toggle;
        harness.ui.set_text_color(if toggle {
            slint::Color::from_rgb_u8(0xf0, 0xa8, 0x30)
        } else {
            slint::Color::from_rgb_u8(0x30, 0xa8, 0xf0)
        });
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("color-update", index, &harness.buffer);
        }
    });

    phase("counter update", FRAMES, |index| {
        counter += 1;
        harness.ui.set_counter(counter);
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("counter-update", index, &harness.buffer);
        }
    });

    phase("late-word update", FRAMES, |index| {
        late_counter += 1;
        harness.ui.set_late_counter(late_counter);
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("late-word-update", index, &harness.buffer);
        }
    });

    phase("mono same-length swap", FRAMES, |index| {
        harness.ui.set_middle_word(words.next().copied().unwrap().into());
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("mono-swap", index, &harness.buffer);
        }
    });

    phase("paragraph width change", FRAMES, |index| {
        toggle = !toggle;
        harness.ui.set_paragraph_width(if toggle { 400. } else { 480. });
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("width-change", index, &harness.buffer);
        }
    });

    phase("elide hidden update", FRAMES, |index| {
        // A change beyond the elision cut: nothing visible moves, and the
        // narrowing finds no difference — today this still repaints the
        // whole element (the no-win case, documented).
        harness.ui.set_hidden_token(if toggle { "hidden-two" } else { "hidden-one" }.into());
        toggle = !toggle;
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("elide-hidden", index, &harness.buffer);
        }
    });

    phase("list-row update", FRAMES, |index| {
        let row = (counter as usize) % 8;
        harness.ui.get_list_rows().set_row_data(
            row,
            format!("S{} {}", counter % 100, if toggle { "up" } else { "down" }).into(),
        );
        counter += 1;
        toggle = !toggle;
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("list-row-update", index, &harness.buffer);
        }
    });

    phase("text-input update", FRAMES, |index| {
        counter += 1;
        harness.ui.set_input_text(format!("PIN {:04}", counter % 10000).into());
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("text-input-update", index, &harness.buffer);
        }
    });

    phase("styled update", FRAMES, |index| {
        counter += 1;
        harness.ui.set_styled_counter(counter);
        assert!(do_rendering(&harness.window, &mut harness.buffer));
        if index < 3 {
            dump.dump("styled-update", index, &harness.buffer);
        }
    });

    phase("counter update (line-buffered)", FRAMES, |index| {
        counter += 1;
        harness.ui.set_counter(counter);
        assert!(harness.window.draw_if_needed(|renderer| {
            let line_buffer = FullBuffer(&mut harness.buffer);
            let _ = renderer.render_by_line(line_buffer);
        }));
        if index < 3 {
            dump.dump("counter-line-buffered", index, &harness.buffer);
        }
    });
}
// The test-side platform shared by every test in this crate: hands out the
// windows each test queued, in order. `set_platform` only succeeds once per
// process; later calls keep the first platform, which reads from the same
// queue, so every test can still hand out its own windows.
#[cfg(test)]
thread_local! {
    static QUEUED_WINDOWS: std::cell::RefCell<Vec<Rc<MinimalSoftwareWindow>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
struct TestPlatform;
#[cfg(test)]
impl slint::platform::Platform for TestPlatform {
    fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, PlatformError> {
        let window = QUEUED_WINDOWS.with(|queue| {
            let mut queue = queue.borrow_mut();
            (!queue.is_empty())
                .then(|| queue.remove(0))
                .expect("queue a window before showing a UI")
        });
        Ok(window)
    }
}

#[cfg(test)]
fn queue_window(window: Rc<MinimalSoftwareWindow>) {
    let _ = slint::platform::set_platform(Box::new(TestPlatform));
    QUEUED_WINDOWS.with(|queue| queue.borrow_mut().push(window));
}
/// The dirty-region narrowing must never leave stale pixels: a frame drawn
/// into a swapped buffer whose dirty region was narrowed by the renderer has
/// to end up byte-identical to a full repaint of the same UI state.
///
/// Two windows render the same UI in lockstep: one with `SwappedBuffers`
/// (partial rendering with the narrowing in place), one with `NewBuffer`
/// (every frame repaints everything). After every property change both
/// buffers must match. Walks counter updates including digit-count changes,
/// late-word changes, monospace same-length and reflow swaps, multibyte
/// changes, line-count changes, elided text, list rows, clipped and styled
/// paragraphs, a text input and an unbroken token, plus the fallback
/// triggers (color, alignment, font size, paragraph geometry) and a moved
/// element.
#[test]
fn narrowed_dirty_region_matches_full_repaint() {
    let _serialized = serialized_test();
    // The window under test: double-buffered with partial rendering.
    let swapped = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::SwappedBuffers,
    );
    queue_window(swapped.clone());
    let ui_narrow = ConcatBench::new().unwrap();
    seed_dynamic_content(&ui_narrow);
    let _ = ui_narrow.show();

    // The reference: every frame repaints everything.
    let full = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::NewBuffer,
    );
    queue_window(full.clone());
    let ui_reference = ConcatBench::new().unwrap();
    seed_dynamic_content(&ui_reference);
    let _ = ui_reference.show();

    let size = slint::PhysicalSize { width: 480, height: 1060 };
    swapped.set_size(size);
    full.set_size(size);

    let draw = |window: &MinimalSoftwareWindow,
                buffer: &mut [Rgb565Pixel]|
     -> (bool, PhysicalRegion) {
        window.request_redraw();
        let mut region = PhysicalRegion::default();
        let drew = window.draw_if_needed(|renderer| {
            region = renderer.render(buffer, size.width as usize);
        });
        (drew, region)
    };

    let mut narrow_buffer = vec![Rgb565Pixel::default(); (size.width * size.height) as usize];
    let mut reference_buffer = vec![Rgb565Pixel::default(); (size.width * size.height) as usize];

    // Applies one change to both UIs and asserts the narrowed repaint stays
    // byte-identical to a full repaint of the same state. Returns the
    // narrowed window's dirty region so visible changes can assert that
    // something actually repainted.
    let change = |name: &str,
                  apply: &dyn Fn(&ConcatBench),
                  narrow: &mut [Rgb565Pixel],
                  reference: &mut [Rgb565Pixel]|
     -> PhysicalRegion {
        apply(&ui_narrow);
        apply(&ui_reference);
        let (drew_narrow, narrow_region) = draw(&swapped, narrow);
        let (drew_reference, _) = draw(&full, reference);
        assert!(drew_narrow, "{name}: nothing to draw");
        assert!(drew_reference, "{name}: nothing to draw");
        assert_eq!(narrow, reference, "{name}: narrowed repaint diverged from a full repaint");
        narrow_region
    };

    // Initial frames: both windows paint everything.
    let (drew_narrow, _) = draw(&swapped, &mut narrow_buffer);
    let (drew_reference, _) = draw(&full, &mut reference_buffer);
    assert!(drew_narrow);
    assert!(drew_reference);
    assert_eq!(narrow_buffer, reference_buffer, "initial frames differ");

    for counter in [8, 9, 10, 11, 99, 100, 101, 1000, 1001] {
        change(
            &format!("counter {counter}"),
            &|ui| ui.set_counter(counter),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // Late-word changes: the dynamic token sits in the last line of the
    // third paragraph, including digit-count changes.
    for late in [5, 9, 10, 11, 99] {
        change(
            &format!("late-word {late}"),
            &|ui| ui.set_late_counter(late),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // Monospace same-length swap: identical advances, only glyph ids change.
    for word in ["gamma", "delta", "theta", "alpha"] {
        change(
            &format!("mono swap {word}"),
            &|ui| ui.set_middle_word(word.into()),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // Reflow: a longer word re-wraps the lines around it.
    for word in ["alphas", "alphabet"] {
        change(
            &format!("mono reflow {word}"),
            &|ui| ui.set_middle_word(word.into()),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // Multibyte changes: the byte offsets of everything after the changed
    // character shift, so the pairing may repaint more than strictly needed
    // — but must never leave stale pixels.
    for word in ["éclair", "eclair"] {
        change(
            &format!("mono multibyte {word}"),
            &|ui| ui.set_middle_word(word.into()),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // A change hidden behind extra lines alters the geometry: the hook is
    // skipped and the old and new element rects are repainted.
    change(
        "line-count +",
        &|ui| ui.set_extra_lines(true),
        &mut narrow_buffer,
        &mut reference_buffer,
    );
    change(
        "line-count -",
        &|ui| ui.set_extra_lines(false),
        &mut narrow_buffer,
        &mut reference_buffer,
    );

    // Alignment and font-size changes repaint the whole element: verify the
    // fallback paths.
    change(
        "alignment change",
        &|ui| ui.set_centered(true),
        &mut narrow_buffer,
        &mut reference_buffer,
    );
    change(
        "alignment back",
        &|ui| ui.set_centered(false),
        &mut narrow_buffer,
        &mut reference_buffer,
    );
    change(
        "font-size change",
        &|ui| ui.set_big_font(true),
        &mut narrow_buffer,
        &mut reference_buffer,
    );
    change(
        "font-size back",
        &|ui| ui.set_big_font(false),
        &mut narrow_buffer,
        &mut reference_buffer,
    );

    // Paragraph geometry: the element gets wider and narrower, re-layouting
    // and repainting its full extent each time.
    for width in [400., 480., 400., 480.] {
        change(
            &format!("width {width}"),
            &|ui| ui.set_paragraph_width(width),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // Moving the element repaints its old and new rects through the moved
    // path, with the narrowing hook skipped.
    change("element moved", &|ui| ui.set_moved(true), &mut narrow_buffer, &mut reference_buffer);
    change(
        "element moved back",
        &|ui| ui.set_moved(false),
        &mut narrow_buffer,
        &mut reference_buffer,
    );

    // Elided text: the counter token sits inside the kept lines (line 2),
    // so a change narrows within the visible lines.
    for elide in [5, 9, 10, 11] {
        change(
            &format!("elide visible {elide}"),
            &|ui| ui.set_elide_counter(elide),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // A change beyond the elision cut: nothing visible moves. The narrowing
    // finds no difference and the element repaints fully — correctness, not
    // efficiency, is what is asserted here.
    change(
        "elide hidden +",
        &|ui| ui.set_hidden_token("hidden-two".into()),
        &mut narrow_buffer,
        &mut reference_buffer,
    );
    change(
        "elide hidden -",
        &|ui| ui.set_hidden_token("hidden-one".into()),
        &mut narrow_buffer,
        &mut reference_buffer,
    );

    // List rows: a model mutation repaints only the row that changed. The
    // models must be seeded: mutating a row of an empty model is a no-op,
    // which would make this whole section pass without testing anything.
    assert_eq!(ui_narrow.get_list_rows().row_count(), 8);
    assert_eq!(ui_reference.get_list_rows().row_count(), 8);
    for row in [0, 5, 3, 7] {
        let value: slint::SharedString = format!("{row} moved-{row}").into();
        let region = change(
            &format!("list row {row}"),
            &move |ui| ui.get_list_rows().set_row_data(row, value.clone()),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
        let size = region.bounding_box_size();
        assert!(
            size.width > 0 && size.height > 0,
            "list row {row}: visible mutation produced an empty dirty region"
        );
    }

    // The clipped element's text: the narrowed region must respect the clip.
    for clipped in [8, 9, 10, 11] {
        change(
            &format!("clipped {clipped}"),
            &|ui| ui.set_counter(clipped),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // The text input: typing repaints the whole input today.
    for pin in ["PIN 1234", "PIN 12345", "PIN 9999"] {
        change(
            &format!("input {pin}"),
            &move |ui| ui.set_input_text(pin.into()),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // The unbroken token: flipping one character deep inside a run without
    // break opportunities goes through the breaker's truncation paths. The
    // flip rides at the start, in the middle and at the end of the run: a
    // change at position 0 shifts every glyph after it.
    for pos in [0, 120, 239] {
        for seed in ["b", "c", "d"] {
            let token = {
                let mut token = "a".repeat(240);
                token.replace_range(pos..pos + 1, seed);
                token
            };
            change(
                &format!("long token {seed} at {pos}"),
                &move |ui| ui.set_long_token(token.clone().into()),
                &mut narrow_buffer,
                &mut reference_buffer,
            );
        }
    }

    // The styled paragraph: the narrowing hook declines styled text (full
    // rect repaint), driven through the markdown interpolation.
    for counter in [12, 13] {
        change(
            &format!("styled {counter}"),
            &|ui| ui.set_styled_counter(counter),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }

    // Color changes repaint the whole element: verify the fallback path.
    change(
        "color change",
        &|ui| ui.set_text_color(slint::Color::from_rgb_u8(0x30, 0xa8, 0xf0)),
        &mut narrow_buffer,
        &mut reference_buffer,
    );
}

/// A text change made while center-aligned must repaint every moved glyph.
/// Center/right alignment shifts a line's origin when its width changes, so a
/// diff that only compares glyph ids and glyph positions relative to the line
/// origin can miss the unchanged glyphs that moved with the line.
#[test]
fn centered_text_change_matches_full_repaint() {
    let _serialized = serialized_test();
    let swapped = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::SwappedBuffers,
    );
    queue_window(swapped.clone());
    let ui_narrow = ConcatBench::new().unwrap();
    let _ = ui_narrow.show();

    let full = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::NewBuffer,
    );
    queue_window(full.clone());
    let ui_reference = ConcatBench::new().unwrap();
    let _ = ui_reference.show();

    let size = slint::PhysicalSize { width: 480, height: 1060 };
    swapped.set_size(size);
    full.set_size(size);

    let draw = |window: &MinimalSoftwareWindow, buffer: &mut [Rgb565Pixel]| {
        window.request_redraw();
        window.draw_if_needed(|renderer| {
            renderer.render(buffer, size.width as usize);
        })
    };

    let mut narrow_buffer = vec![Rgb565Pixel::default(); (size.width * size.height) as usize];
    let mut reference_buffer = vec![Rgb565Pixel::default(); (size.width * size.height) as usize];

    let change = |name: &str,
                  apply: &dyn Fn(&ConcatBench),
                  narrow: &mut [Rgb565Pixel],
                  reference: &mut [Rgb565Pixel]| {
        apply(&ui_narrow);
        apply(&ui_reference);
        assert!(draw(&swapped, narrow), "{name}: nothing to draw");
        assert!(draw(&full, reference), "{name}: nothing to draw");
        assert_eq!(narrow, reference, "{name}: narrowed repaint diverged from a full repaint");
    };

    assert!(draw(&swapped, &mut narrow_buffer));
    assert!(draw(&full, &mut reference_buffer));
    assert_eq!(narrow_buffer, reference_buffer, "initial frames differ");

    // Settle the centered layout through its full-element fallback, then keep
    // the alignment fixed while the text itself changes.
    change("centered on", &|ui| ui.set_centered(true), &mut narrow_buffer, &mut reference_buffer);
    for counter in [8, 9, 10, 11, 99, 100, 101, 1000, 1001] {
        change(
            &format!("centered counter {counter}"),
            &|ui| ui.set_counter(counter),
            &mut narrow_buffer,
            &mut reference_buffer,
        );
    }
}

/// Emptying a text must not leave a stale layout behind: the empty frame
/// draws nothing and stores nothing, so the next text change must not diff
/// against the layout cached before the text disappeared.
///
/// This runs on a reused buffer on purpose: swapped buffers union the
/// previous frame's dirty region into every frame, which would repaint the
/// whole item anyway and hide the stale base. A reused buffer has no such
/// union, so a diff against the pre-empty layout leaves the shared prefix
/// stale.
#[test]
fn empty_text_transition_matches_full_repaint() {
    let _serialized = serialized_test();
    let reused = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::ReusedBuffer,
    );
    queue_window(reused.clone());
    let ui_narrow = ConcatBench::new().unwrap();
    let _ = ui_narrow.show();

    let full = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::NewBuffer,
    );
    queue_window(full.clone());
    let ui_reference = ConcatBench::new().unwrap();
    let _ = ui_reference.show();

    let size = slint::PhysicalSize { width: 480, height: 1060 };
    reused.set_size(size);
    full.set_size(size);

    let draw = |window: &MinimalSoftwareWindow, buffer: &mut [Rgb565Pixel]| {
        window.request_redraw();
        window.draw_if_needed(|renderer| {
            renderer.render(buffer, size.width as usize);
        })
    };

    let mut narrow_buffer = vec![Rgb565Pixel::default(); (size.width * size.height) as usize];
    let mut reference_buffer = vec![Rgb565Pixel::default(); (size.width * size.height) as usize];

    let change = |name: &str,
                  apply: &dyn Fn(&ConcatBench),
                  narrow: &mut [Rgb565Pixel],
                  reference: &mut [Rgb565Pixel]| {
        apply(&ui_narrow);
        apply(&ui_reference);
        assert!(draw(&reused, narrow), "{name}: nothing to draw");
        assert!(draw(&full, reference), "{name}: nothing to draw");
        assert_eq!(narrow, reference, "{name}: narrowed repaint diverged from a full repaint");
    };

    assert!(draw(&reused, &mut narrow_buffer));
    assert!(draw(&full, &mut reference_buffer));
    assert_eq!(narrow_buffer, reference_buffer, "initial frames differ");

    change("counter 8", &|ui| ui.set_counter(8), &mut narrow_buffer, &mut reference_buffer);
    let before_empty = narrow_buffer.clone();
    change(
        "counter emptied",
        &|ui| ui.set_empty_counter(true),
        &mut narrow_buffer,
        &mut reference_buffer,
    );
    // The empty frame must actually clear the text: otherwise the restore
    // below would diff against a still-visible layout and prove nothing.
    assert_ne!(
        narrow_buffer, before_empty,
        "emptying the counter text changed no pixels"
    );
    change(
        "counter restored",
        &|ui| {
            ui.set_empty_counter(false);
            ui.set_counter(9);
        },
        &mut narrow_buffer,
        &mut reference_buffer,
    );
}

/// A narrowed window and a full-repaint reference rendering the same UI in
/// lockstep, for the pixel-equality tests below.
#[cfg(test)]
struct Lockstep {
    narrow: Rc<MinimalSoftwareWindow>,
    reference: Rc<MinimalSoftwareWindow>,
    ui_narrow: ConcatBench,
    ui_reference: ConcatBench,
    narrow_buffer: Vec<Rgb565Pixel>,
    reference_buffer: Vec<Rgb565Pixel>,
    size: slint::PhysicalSize,
}

#[cfg(test)]
impl Lockstep {
    fn new(buffer_type: RepaintBufferType) -> Self {
        let narrow = MinimalSoftwareWindow::new(buffer_type);
        queue_window(narrow.clone());
        let ui_narrow = ConcatBench::new().unwrap();
        let _ = ui_narrow.show();

        let reference = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
        queue_window(reference.clone());
        let ui_reference = ConcatBench::new().unwrap();
        let _ = ui_reference.show();

        let size = slint::PhysicalSize { width: 480, height: 1060 };
        narrow.set_size(size);
        reference.set_size(size);
        let pixels = (size.width * size.height) as usize;
        let mut lockstep = Self {
            narrow,
            reference,
            ui_narrow,
            ui_reference,
            narrow_buffer: vec![Rgb565Pixel::default(); pixels],
            reference_buffer: vec![Rgb565Pixel::default(); pixels],
            size,
        };
        assert!(lockstep.draw_narrow(), "initial narrow frame drew nothing");
        assert!(lockstep.draw_reference(), "initial reference frame drew nothing");
        assert_eq!(
            lockstep.narrow_buffer, lockstep.reference_buffer,
            "initial frames differ"
        );
        lockstep
    }

    fn draw_narrow(&mut self) -> bool {
        self.narrow.request_redraw();
        self.narrow.draw_if_needed(|renderer| {
            renderer.render(&mut self.narrow_buffer, self.size.width as usize);
        })
    }

    fn draw_reference(&mut self) -> bool {
        self.reference.request_redraw();
        self.reference.draw_if_needed(|renderer| {
            renderer.render(&mut self.reference_buffer, self.size.width as usize);
        })
    }

    /// Applies one change to both UIs and asserts the narrowed repaint stays
    /// byte-identical to a full repaint of the same state.
    fn change(&mut self, name: &str, apply: &dyn Fn(&ConcatBench)) {
        apply(&self.ui_narrow);
        apply(&self.ui_reference);
        assert!(self.draw_narrow(), "{name}: nothing to draw");
        assert!(self.draw_reference(), "{name}: nothing to draw");
        assert_eq!(
            self.narrow_buffer, self.reference_buffer,
            "{name}: narrowed repaint diverged from a full repaint"
        );
    }

    /// Like [`Self::change`], but returns the narrowed window's dirty region
    /// so quality tests can assert on its size and position.
    fn change_region(&mut self, name: &str, apply: &dyn Fn(&ConcatBench)) -> PhysicalRegion {
        apply(&self.ui_narrow);
        apply(&self.ui_reference);
        self.narrow.request_redraw();
        let mut region = PhysicalRegion::default();
        assert!(
            self.narrow.draw_if_needed(|renderer| {
                region = renderer.render(&mut self.narrow_buffer, self.size.width as usize);
            }),
            "{name}: nothing to draw"
        );
        assert!(self.draw_reference(), "{name}: nothing to draw");
        assert_eq!(
            self.narrow_buffer, self.reference_buffer,
            "{name}: narrowed repaint diverged from a full repaint"
        );
        region
    }
}

/// Sweep for the single-line alignment tests: every step changes the width
/// (including both digit-count directions) plus a multibyte pair, and no two
/// consecutive entries are equal.
#[cfg(test)]
const ALIGN_SWEEP_UP: &[&str] = &[
    "Value 8",
    "Value 9",
    "Value 10",
    "Value 11",
    "Value 98",
    "Value 99",
    "Value 100",
    "Value 101",
    "Value 999",
    "Value 1000",
    "Value 1001",
    "Value café 7",
    "Value cafe 7",
];

/// Center-aligned single-line text must repaint every moved glyph: the box
/// has a fixed width, so each sweep step shifts the line origin without
/// re-wrapping, and the diff must compare it against the previous layout.
#[test]
fn centered_single_line_matches_full_repaint() {
    let _serialized = serialized_test();
    let mut lockstep = Lockstep::new(RepaintBufferType::SwappedBuffers);
    lockstep.change("align center", &|ui| ui.set_align_test_mode(1));
    for value in ALIGN_SWEEP_UP {
        lockstep.change(
            &format!("centered {value}"),
            &|ui| ui.set_align_test_text((*value).into()),
        );
    }
    for value in ALIGN_SWEEP_UP[..ALIGN_SWEEP_UP.len() - 1].iter().rev() {
        lockstep.change(
            &format!("centered back {value}"),
            &|ui| ui.set_align_test_text((*value).into()),
        );
    }
}

/// Right alignment shifts line origins exactly like center does, from the
/// other side; it has no coverage without this test.
#[test]
fn right_aligned_single_line_matches_full_repaint() {
    let _serialized = serialized_test();
    let mut lockstep = Lockstep::new(RepaintBufferType::SwappedBuffers);
    lockstep.change("align right", &|ui| ui.set_align_test_mode(2));
    for value in ALIGN_SWEEP_UP {
        lockstep.change(
            &format!("right {value}"),
            &|ui| ui.set_align_test_text((*value).into()),
        );
    }
    for value in ALIGN_SWEEP_UP[..ALIGN_SWEEP_UP.len() - 1].iter().rev() {
        lockstep.change(
            &format!("right back {value}"),
            &|ui| ui.set_align_test_text((*value).into()),
        );
    }
}

/// Changing the alignment and the text in the same frame must stay correct:
/// the alignment change alters the layout inputs, so the hook has to decline
/// and repaint the whole element.
#[test]
fn alignment_and_text_same_frame_matches_full_repaint() {
    let _serialized = serialized_test();
    let mut lockstep = Lockstep::new(RepaintBufferType::SwappedBuffers);
    for (mode, value) in [(1, "Value 8"), (2, "Value 9"), (0, "Value 10"), (1, "Value 11")] {
        lockstep.change(
            &format!("align {mode} with {value}"),
            &|ui| {
                ui.set_align_test_mode(mode);
                ui.set_align_test_text(value.into());
            },
        );
    }
}

/// Changing a paragraph's width while it is centered must repaint the old
/// and new element rects through the geometry fallback, like it does
/// left-aligned.
#[test]
fn width_change_while_centered_matches_full_repaint() {
    let _serialized = serialized_test();
    let mut lockstep = Lockstep::new(RepaintBufferType::SwappedBuffers);
    lockstep.change("centered on", &|ui| ui.set_centered(true));
    for width in [400., 480., 400.] {
        lockstep.change(
            &format!("centered width {width}"),
            &|ui| ui.set_paragraph_width(width),
        );
    }
}

/// Short strings in Hebrew, Arabic, Thai and CJK mono must narrow exactly
/// like Latin ones. The builtin layout has no bidi engine (logical order
/// renders left to right), so these pin narrow==full equivalence per
/// script, never visual correctness. Runs on both buffer types: swapped
/// buffers union the previous frame's region and would hide a stale base
/// that reused buffers expose.
#[test]
fn script_text_matches_full_repaint() {
    let _serialized = serialized_test();
    let mut swapped = Lockstep::new(RepaintBufferType::SwappedBuffers);
    let mut reused = Lockstep::new(RepaintBufferType::ReusedBuffer);
    let mut group = |mode: i32, name: &str, values: &[&str]| {
        let setup = format!("{name} font");
        swapped.change(&setup, &|ui| ui.set_script_mode(mode));
        reused.change(&setup, &|ui| ui.set_script_mode(mode));
        for value in values {
            let step = format!("{name} {value}");
            swapped.change(&step, &|ui| ui.set_script_text((*value).into()));
            reused.change(&step, &|ui| ui.set_script_text((*value).into()));
        }
    };
    // Different words (hence widths) so line origins shift under
    // center/right alignment the same way the Latin sweeps shift them.
    group(0, "hebrew", &["שלום", "עולם", "אב", "שלום רב", "שלום"]);
    group(1, "arabic", &["مرحبا", "مرحبا 7", "مرحبا 10", "مرحبا 100", "العالم", "مرحبا"]);
    group(2, "thai", &["สวัสดี", "ดี", "สวัสดีครับ", "สวัสดี"]);
    group(3, "cjk", &["日本語", "日本語 7", "日本語 10", "中国語", "abc 日本語 def", "日本語"]);
}

/// Width changes under center/right alignment must narrow per script, not
/// just in Latin: each step shifts the line origin without re-wrapping, and
/// a missed origin shift leaves stale pixels. Runs on both buffer types.
#[test]
fn script_alignment_matches_full_repaint() {
    let _serialized = serialized_test();
    let mut swapped = Lockstep::new(RepaintBufferType::SwappedBuffers);
    let mut reused = Lockstep::new(RepaintBufferType::ReusedBuffer);
    let groups: [(i32, &str, &[&str]); 4] = [
        (0, "hebrew", &["שלום", "עולם", "אב", "שלום רב", "שלום"]),
        (1, "arabic", &["مرحبا", "مرحبا 7", "مرحبا 100", "العالم", "مرحبا"]),
        (2, "thai", &["สวัสดี", "ดี", "สวัสดีครับ", "สวัสดี"]),
        (3, "cjk", &["日本語", "日本語 7", "日本語 100", "中国語", "日本語"]),
    ];
    for align in [1, 2] {
        for (mode, name, values) in &groups {
            let setup = format!("{name} align {align}");
            swapped.change(&setup, &|ui| {
                ui.set_script_mode(*mode);
                ui.set_script_align_mode(align);
            });
            reused.change(&setup, &|ui| {
                ui.set_script_mode(*mode);
                ui.set_script_align_mode(align);
            });
            for value in *values {
                let step = format!("{name} align {align} {value}");
                swapped.change(&step, &|ui| ui.set_script_text((*value).into()));
                reused.change(&step, &|ui| ui.set_script_text((*value).into()));
            }
        }
    }
}

/// A same-count CJK swap on the monospace CJK face changes only glyph ids
/// with identical advances and positions: the dirty region must stay
/// word-sized and sit at the line start, like the Latin mono swap.
#[test]
fn cjk_same_width_swap_stays_tight() {
    let _serialized = serialized_test();
    let mut lockstep = Lockstep::new(RepaintBufferType::SwappedBuffers);
    lockstep.change("cjk font", &|ui| {
        ui.set_script_mode(3);
        ui.set_script_text("日本語".into());
    });
    // The first swap unions the previous full-item region, like the mono
    // tight test's settling swap; the steady-state swap is measured.
    lockstep.change("cjk settle", &|ui| ui.set_script_text("中国語".into()));
    let region = lockstep.change_region("cjk swap", &|ui| ui.set_script_text("日本語".into()));
    let (origin, size) = (region.bounding_box_origin(), region.bounding_box_size());
    assert!(
        (300..320).contains(&origin.x) && (845..865).contains(&origin.y),
        "cjk swap region origin {origin:?} is not at the line start",
    );
    assert!(
        size.width <= 120 && size.height <= 60,
        "cjk swap region {}x{} is not word-sized",
        size.width,
        size.height,
    );
}

/// A same-width first-word swap must repaint only the word at the line
/// start: the static tail never moves in the source string, and with equal
/// advances it must not move on screen either.
#[test]
fn first_word_same_width_swap_stays_tight() {
    let _serialized = serialized_test();
    let mut lockstep = Lockstep::new(RepaintBufferType::SwappedBuffers);
    let assert_word_sized = |region: &PhysicalRegion, name: &str| {
        let (origin, size) = (region.bounding_box_origin(), region.bounding_box_size());
        assert!(
            (300..320).contains(&origin.x) && (800..820).contains(&origin.y),
            "{name} region origin {origin:?} is not at the line start",
        );
        assert!(
            size.width <= 120 && size.height <= 60,
            "{name} region {}x{} is not word-sized",
            size.width,
            size.height,
        );
    };
    lockstep.change("lead gamma", &|ui| ui.set_lead_word("gamma".into()));
    let region = lockstep.change_region("lead delta", &|ui| ui.set_lead_word("delta".into()));
    assert_word_sized(&region, "lead delta");
    // Greek would be ideal here, but the vendored mono subset has no Greek
    // glyphs at all (missing glyphs consume a fixed invisible advance, so a
    // Greek swap narrows to nothing and falls back) — a second Latin pair
    // instead.
    lockstep.change("lead theta", &|ui| ui.set_lead_word("theta".into()));
    let region = lockstep.change_region("lead alpha", &|ui| ui.set_lead_word("alpha".into()));
    assert_word_sized(&region, "lead alpha");
}

/// A different-width first word shifts the static tail on screen: the
/// repaint must cover the word and everything it pushed sideways, and match
/// a full repaint exactly. Runs on both buffer types.
#[test]
fn first_word_width_change_matches_full_repaint() {
    let _serialized = serialized_test();
    let mut swapped = Lockstep::new(RepaintBufferType::SwappedBuffers);
    let mut reused = Lockstep::new(RepaintBufferType::ReusedBuffer);
    for word in ["abc", "alphabet", "alphabet soup", "abc", "alpha"] {
        let name = format!("lead {word}");
        swapped.change(&name, &|ui| ui.set_lead_word((*word).into()));
        reused.change(&name, &|ui| ui.set_lead_word((*word).into()));
    }
}

/// Deterministic xorshift64star: the fuzz test needs no rand dependency and
/// replays the exact same cases on every run.
#[cfg(test)]
struct FuzzRng(u64);

#[cfg(test)]
impl FuzzRng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Fragment pools per fuzz-mode: words, digits, spaces and newlines drawn
/// from scripts the mode's font really covers (see the vendored fonts), plus
/// the combining/ZWJ/emoji paths on the mono face. Fragments are multi-char
/// slices so byte offsets shift naturally under edits.
#[cfg(test)]
const FUZZ_HEBREW: &[&str] = &["שלום", "עולם", "אב", "רב", " ", "  ", "\n", "7", "100"];
#[cfg(test)]
const FUZZ_ARABIC: &[&str] = &["مرحبا", "العالم", " ", "\n", "9", "42", "  "];
#[cfg(test)]
const FUZZ_THAI: &[&str] = &["สวัสดี", "ดี", "ครับ", " ", "\n", "  "];
#[cfg(test)]
const FUZZ_CJK: &[&str] = &["日本語", "中国語", "テスト", "abc", "def", " ", "\n", "7", "100"];
#[cfg(test)]
const FUZZ_MONO: &[&str] = &[
    "alpha",
    "gamma",
    "café",
    "cafe\u{301}",
    "a\u{200d}b",
    "😊",
    " ",
    "  ",
    "\n",
    "…",
    "●",
    "9",
    "xyz",
];

/// Random mixed-script edits must narrow exactly like hand-written ones: 300
/// deterministic cases (fixed seed replays identically) of substitutions,
/// insertions and deletions at fragment boundaries, plus alignment changes
/// that shift line origins, on both buffer types.
/// The failure message carries the case, mode and text — not megabytes of
/// pixels — so a red run is directly reproducible.
#[test]
fn fuzz_text_matches_full_repaint() {
    let _serialized = serialized_test();
    let mut swapped = Lockstep::new(RepaintBufferType::SwappedBuffers);
    let mut reused = Lockstep::new(RepaintBufferType::ReusedBuffer);
    let pools = [FUZZ_HEBREW, FUZZ_ARABIC, FUZZ_THAI, FUZZ_CJK, FUZZ_MONO];
    let mut rng = FuzzRng(0x12345678);
    let mut fragments: Vec<&str> = vec!["fuzz"];
    let mut mode = 4usize;
    let mut align = 0usize;
    let mut last = (mode, align, fragments.concat());
    for case in 0..300 {
        if rng.below(15) == 0 {
            mode = rng.below(pools.len());
            if rng.below(3) == 0 {
                // Width changes shift line origins under center/right, the
                // same path the script×alignment test pins deterministically.
                align = rng.below(3);
            }
        } else if rng.below(30) == 0 {
            // Alignment-only change: same text, moved line origins.
            align = rng.below(3);
        } else {
            for _ in 0..1 + rng.below(3) {
                let pool = pools[mode];
                match rng.below(4) {
                    0 if !fragments.is_empty() => {
                        fragments.remove(rng.below(fragments.len()));
                    }
                    1 => {
                        fragments.insert(rng.below(fragments.len() + 1), pool[rng.below(pool.len())]);
                    }
                    _ if !fragments.is_empty() => {
                        let i = rng.below(fragments.len());
                        fragments[i] = pool[rng.below(pool.len())];
                    }
                    _ => {}
                }
            }
            while fragments.iter().map(|fragment| fragment.chars().count()).sum::<usize>() > 120
                && !fragments.is_empty()
            {
                fragments.remove(rng.below(fragments.len()));
            }
        }
        let text = fragments.concat();
        if (mode, align, text.clone()) == last {
            // Setting identical properties draws nothing; skip the case
            // rather than tripping the drew assertions.
            continue;
        }
        last = (mode, align, text.clone());
        let apply = |ui: &ConcatBench| {
            ui.set_fuzz_mode(mode as i32);
            ui.set_fuzz_align_mode(align as i32);
            ui.set_fuzz_text(text.clone().into());
        };
        apply(&swapped.ui_narrow);
        apply(&swapped.ui_reference);
        assert!(swapped.draw_narrow(), "fuzz {case}: narrow drew nothing");
        assert!(swapped.draw_reference(), "fuzz {case}: reference drew nothing");
        assert!(
            swapped.narrow_buffer == swapped.reference_buffer,
            "fuzz {case} diverged on swapped buffers for mode {mode} align {align} text {text:?}"
        );
        apply(&reused.ui_narrow);
        apply(&reused.ui_reference);
        assert!(reused.draw_narrow(), "fuzz {case}: narrow drew nothing");
        assert!(reused.draw_reference(), "fuzz {case}: reference drew nothing");
        assert!(
            reused.narrow_buffer == reused.reference_buffer,
            "fuzz {case} diverged on reused buffers for mode {mode} align {align} text {text:?}"
        );
    }
}

/// Every layout input outside the string must decline narrowing and repaint
/// the whole element: letter spacing, line height, wrap mode and vertical
/// alignment all sit in the cache key. Each toggle must pass before and
/// after any renderer change.
#[test]
fn layout_input_fallbacks_match_full_repaint() {
    let _serialized = serialized_test();
    let mut lockstep = Lockstep::new(RepaintBufferType::SwappedBuffers);
    lockstep.change("spacing on", &|ui| ui.set_extra_spacing(true));
    lockstep.change("spacing off", &|ui| ui.set_extra_spacing(false));
    lockstep.change("tall lines on", &|ui| ui.set_tall_lines(true));
    lockstep.change("tall lines off", &|ui| ui.set_tall_lines(false));
    lockstep.change("elide fuzz on", &|ui| ui.set_elide_fuzz(true));
    lockstep.change("elide fuzz off", &|ui| ui.set_elide_fuzz(false));
    for mode in [1, 2, 0] {
        lockstep.change(
            &format!("valign {mode}"),
            &|ui| ui.set_valign_mode(mode),
        );
    }
}

/// Pins the *quality* of the narrowed dirty regions, not just their
/// correctness: a redraw of a same-length monospace word must stay within
/// roughly the word's extent, and a change in the last line of a paragraph
/// must stay within roughly that line. A regression that repaints whole
/// lines or paragraphs shows up as a fat bounding box.
#[test]
fn narrowed_dirty_regions_stay_tight() {
    let _serialized = serialized_test();
    let swapped = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::SwappedBuffers,
    );
    queue_window(swapped.clone());
    let ui = ConcatBench::new().unwrap();
    let _ = ui.show();
    swapped.set_size(SIZE);

    let mut buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];
    let draw = |buffer: &mut [Rgb565Pixel]| {
        let mut region = None;
        swapped.request_redraw();
        swapped.draw_if_needed(|renderer| {
            region = Some(renderer.render(buffer, SIZE.width as usize));
        });
        region
    };

    // Prime the scene.
    assert!(draw(&mut buffer).is_some());

    let mut same_length_word = |word: &str| -> (f32, f32) {
        ui.set_middle_word(word.into());
        let region = draw(&mut buffer).expect("mono swap must dirty something");
        let size = region.bounding_box_size();
        (size.width as f32, size.height as f32)
    };

    // The first change frame unions the previous full-window region, so a
    // settling swap comes first; what is measured is steady-state.
    let _ = same_length_word("delta");

    // The same-length monospace swap: the changed word is about five 8px
    // glyphs on a ~20px line box, plus raster extents; anything much larger
    // means the narrowing regressed to whole lines or the element.
    let (swap_width, swap_height) = same_length_word("gamma");
    assert!(
        swap_width <= 120. && swap_height <= 60.,
        "same-length swap region {swap_width}x{swap_height} is not word-sized"
    );
    let _ = same_length_word("theta");

    // The late-word change: only the last line may be dirty. Settle first
    // for the same reason as above.
    ui.set_late_counter(41);
    let _ = draw(&mut buffer);
    ui.set_late_counter(42);
    let region = draw(&mut buffer).expect("late-word change must dirty something");
    let (late_width, late_height) =
        (region.bounding_box_size().width as f32, region.bounding_box_size().height as f32);
    assert!(
        late_width <= 480. && late_height <= 60.,
        "late-word region {late_width}x{late_height} is not last-line-sized"
    );
}

/// The line-buffered rasterization path must paint the same pixels as the
/// full-buffer path: rendering the narrowed frame line by line stays
/// byte-identical to a full repaint.
#[test]
fn narrowed_render_by_line_matches_full_repaint() {
    let _serialized = serialized_test();
    let swapped = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::SwappedBuffers,
    );
    queue_window(swapped.clone());
    let ui_narrow = ConcatBench::new().unwrap();
    let _ = ui_narrow.show();

    let full = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::NewBuffer,
    );
    queue_window(full.clone());
    let ui_reference = ConcatBench::new().unwrap();
    let _ = ui_reference.show();

    swapped.set_size(SIZE);
    full.set_size(SIZE);

    let draw_narrow_by_line = |buffer: &mut [Rgb565Pixel]| {
        let mut region = None;
        swapped.request_redraw();
        swapped.draw_if_needed(|renderer| {
            region = Some(renderer.render_by_line(FullBuffer(buffer)));
        });
        region
    };
    let draw_full = |buffer: &mut [Rgb565Pixel]| {
        let mut region = None;
        full.request_redraw();
        full.draw_if_needed(|renderer| {
            region = Some(renderer.render(buffer, SIZE.width as usize));
        });
        region
    };

    let mut narrow_buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];
    let mut reference_buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];

    assert!(draw_narrow_by_line(&mut narrow_buffer).is_some());
    assert!(draw_full(&mut reference_buffer).is_some());
    assert_eq!(narrow_buffer, reference_buffer, "initial line-buffered frame differs");

    for counter in [8, 9, 10, 11, 12] {
        ui_narrow.set_counter(counter);
        ui_reference.set_counter(counter);
        assert!(draw_narrow_by_line(&mut narrow_buffer).is_some());
        assert!(draw_full(&mut reference_buffer).is_some());
        assert_eq!(
            narrow_buffer, reference_buffer,
            "counter {counter}: line-buffered narrowed repaint diverged"
        );
    }
}

/// A rotated screen must narrow and repaint the same way, whatever the
/// rotation and buffer type: rotation maps the logical dirty rects onto the
/// rotated buffer, and the result must stay byte-identical to a full repaint.
#[test]
fn narrowed_matches_full_repaint_rotated() {
    let _serialized = serialized_test();
    use RenderingRotation::{Rotate180, Rotate270, Rotate90};
    use RepaintBufferType::{ReusedBuffer, SwappedBuffers};
    for rotation in [Rotate90, Rotate180, Rotate270] {
        for buffer_type in [SwappedBuffers, ReusedBuffer] {
            check_rotated(rotation, buffer_type);
        }
    }
}

#[cfg(test)]
fn check_rotated(rotation: RenderingRotation, buffer_type: RepaintBufferType) {
    let narrow = MinimalSoftwareWindow::new(buffer_type);
    queue_window(narrow.clone());
    let ui_narrow = ConcatBench::new().unwrap();
    let _ = ui_narrow.show();

    let reference = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    queue_window(reference.clone());
    let ui_reference = ConcatBench::new().unwrap();
    let _ = ui_reference.show();

    narrow.set_size(SIZE);
    reference.set_size(SIZE);

    // Rotated rendering writes transposed for 90/270: the buffer is
    // per-pixel-row addressed along the rotated stride.
    let stride = match rotation {
        RenderingRotation::Rotate90 | RenderingRotation::Rotate270 => SIZE.height,
        _ => SIZE.width,
    } as usize;
    let name = |what: &str| format!("{what} {rotation:?} {buffer_type:?}");
    let draw = |window: &MinimalSoftwareWindow, buffer: &mut [Rgb565Pixel]| {
        let mut region = None;
        window.request_redraw();
        window.draw_if_needed(|renderer| {
            renderer.set_rendering_rotation(rotation);
            region = Some(renderer.render(buffer, stride));
        });
        region
    };

    let mut narrow_buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];
    let mut reference_buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];

    // The first rotated frame settles the rotation on the renderer.
    assert!(draw(&narrow, &mut narrow_buffer).is_some());
    assert!(draw(&reference, &mut reference_buffer).is_some());
    assert_eq!(
        narrow_buffer,
        reference_buffer,
        "{}: initial rotated frames differ",
        name("initial")
    );

    for counter in [8, 9, 10, 11] {
        ui_narrow.set_counter(counter);
        ui_reference.set_counter(counter);
        assert!(draw(&narrow, &mut narrow_buffer).is_some());
        assert!(draw(&reference, &mut reference_buffer).is_some());
        assert_eq!(
            narrow_buffer, reference_buffer,
            "{} counter {counter}: rotated narrowed repaint diverged",
            name("rotated")
        );
    }
}

/// Line-buffered rendering under rotation must match a full repaint too:
/// the narrowed frame rendered line by line in rotated space stays
/// byte-identical to a buffered full repaint. (This tripped a line-walker
/// `debug_assert` before the span-expiry fix below this commit's base.)
#[test]
fn narrowed_render_by_line_matches_full_repaint_rotated() {
    let _serialized = serialized_test();
    let narrow = MinimalSoftwareWindow::new(RepaintBufferType::SwappedBuffers);
    queue_window(narrow.clone());
    let ui_narrow = ConcatBench::new().unwrap();
    let _ = ui_narrow.show();

    let reference = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    queue_window(reference.clone());
    let ui_reference = ConcatBench::new().unwrap();
    let _ = ui_reference.show();

    narrow.set_size(SIZE);
    reference.set_size(SIZE);

    /// Transposed view over the frame buffer: under Rotate90 each rendered
    /// line is SIZE.height pixels addressed along the rotated stride.
    struct RotatedFullBuffer<'a>(&'a mut [Rgb565Pixel]);

    impl LineBufferProvider for RotatedFullBuffer<'_> {
        type TargetPixel = Rgb565Pixel;
        fn process_line(
            &mut self,
            line: usize,
            range: core::ops::Range<usize>,
            render_fn: impl FnOnce(&mut [Self::TargetPixel]),
        ) {
            render_fn(&mut self.0[line * SIZE.height as usize..][range]);
        }
    }

    let draw_narrow_by_line = |buffer: &mut [Rgb565Pixel]| {
        let mut region = None;
        narrow.request_redraw();
        narrow.draw_if_needed(|renderer| {
            renderer.set_rendering_rotation(RenderingRotation::Rotate90);
            region = Some(renderer.render_by_line(RotatedFullBuffer(buffer)));
        });
        region
    };
    let draw_full = |buffer: &mut [Rgb565Pixel]| {
        let mut region = None;
        reference.request_redraw();
        reference.draw_if_needed(|renderer| {
            renderer.set_rendering_rotation(RenderingRotation::Rotate90);
            region = Some(renderer.render(buffer, SIZE.height as usize));
        });
        region
    };

    let mut narrow_buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];
    let mut reference_buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];

    assert!(draw_narrow_by_line(&mut narrow_buffer).is_some());
    assert!(draw_full(&mut reference_buffer).is_some());
    assert_eq!(narrow_buffer, reference_buffer, "initial rotated line-buffered frame differs");

    for counter in [8, 9, 10, 11, 99, 100, 101] {
        ui_narrow.set_counter(counter);
        ui_reference.set_counter(counter);
        assert!(draw_narrow_by_line(&mut narrow_buffer).is_some());
        assert!(draw_full(&mut reference_buffer).is_some());
        assert_eq!(
            narrow_buffer, reference_buffer,
            "counter {counter}: rotated line-buffered narrowed repaint diverged"
        );
    }
}

/// Text under an item transform must narrow like untransformed text: a
/// non-right-angle rotation sends the narrowed rect through the generic
/// transformed-rect path, and the repaint must still match a full repaint
/// exactly.
#[test]
fn transformed_text_matches_full_repaint() {
    let _serialized = serialized_test();
    let narrow = MinimalSoftwareWindow::new(RepaintBufferType::SwappedBuffers);
    queue_window(narrow.clone());
    let ui_narrow = TransformBench::new().unwrap();
    let _ = ui_narrow.show();

    let reference = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    queue_window(reference.clone());
    let ui_reference = TransformBench::new().unwrap();
    let _ = ui_reference.show();

    let size = slint::PhysicalSize { width: 480, height: 200 };
    narrow.set_size(size);
    reference.set_size(size);

    let draw = |window: &MinimalSoftwareWindow, buffer: &mut [Rgb565Pixel]| {
        window.request_redraw();
        window.draw_if_needed(|renderer| {
            renderer.render(buffer, size.width as usize);
        })
    };

    let mut narrow_buffer = vec![Rgb565Pixel::default(); (size.width * size.height) as usize];
    let mut reference_buffer = vec![Rgb565Pixel::default(); (size.width * size.height) as usize];

    assert!(draw(&narrow, &mut narrow_buffer));
    assert!(draw(&reference, &mut reference_buffer));
    assert_eq!(narrow_buffer, reference_buffer, "initial transformed frames differ");

    for counter in [8, 9, 10, 11, 99, 100, 101] {
        ui_narrow.set_counter(counter);
        ui_reference.set_counter(counter);
        assert!(draw(&narrow, &mut narrow_buffer), "counter {counter}: nothing to draw");
        assert!(draw(&reference, &mut reference_buffer), "counter {counter}: nothing to draw");
        assert_eq!(
            narrow_buffer, reference_buffer,
            "counter {counter}: transformed narrowed repaint diverged"
        );
    }
}

/// Every frame's allocation profile must stay bounded: unchanged redraws are
/// deterministic and retain no memory, cheap text changes allocate within a
/// fixed budget over the unchanged baseline, and no phase grows the retained
/// bytes from frame to frame. The tests in this crate hold a process-wide
/// serial guard, so no other test in this crate allocates concurrently with a
/// measurement. The dirty-region walk
/// itself allocates per frame (pre-existing behavior) — what is asserted is
/// that it neither grows nor leaks, and that the narrowing adds no
/// meaningful allocations on top.
#[test]
fn frames_stay_allocation_bounded() {
    let _serialized = serialized_test();
    // One shared window and UI, straight through the bench platform.
    let window = MinimalSoftwareWindow::new(
        slint::platform::software_renderer::RepaintBufferType::SwappedBuffers,
    );
    queue_window(window.clone());
    let ui = ConcatBench::new().unwrap();
    let _ = ui.show();
    window.set_size(SIZE);
    let mut buffer = vec![Rgb565Pixel::default(); (SIZE.width * SIZE.height) as usize];

    let do_frame = |buffer: &mut [Rgb565Pixel]| -> (u64, i64) {
        let before = AllocSnapshot::take();
        window.request_redraw();
        window.draw_if_needed(|renderer| {
            let _ = renderer.render(buffer, SIZE.width as usize);
        });
        AllocSnapshot::take().delta_since(&before)
    };

    // Prime the scene (this frame allocates and is not measured). The frame
    // after it still repaints the full window (the swapped-buffer union with
    // the prime frame's region), so warm up one more frame before measuring:
    // only then are the frames true no-ops.
    assert!(do_frame(&mut buffer).0 > 0);
    assert!(do_frame(&mut buffer).0 > 0);

    // Unchanged redraws: allocations stay within the walk's machinery
    // budget, and the net retained bytes settle to zero — the walk's
    // temporary allocations all free. (Exact counts drift by a few units
    // with shared font-cache warm state, so they are capped, not compared;
    // one-time cache-settling frees can land in either frame, so the net
    // is bounded rather than zero — what must never happen is recurring
    // growth.)
    let (first_allocs, first_net) = do_frame(&mut buffer);
    let (second_allocs, second_net) = do_frame(&mut buffer);
    assert!(
        first_allocs <= 2048 && second_allocs <= 2048,
        "unchanged frames allocate {first_allocs}/{second_allocs} times"
    );
    assert!(
        first_net.abs() <= 1024 * 1024 && second_net.abs() <= 1024 * 1024,
        "unchanged frames retained {first_net}/{second_net} bytes"
    );

    // Cheap text changes: each swing allocates within a fixed budget over
    // the unchanged baseline. A layout cache, when present, retains a
    // bounded number of entries — what must hold everywhere is that
    // repeating the same change neither accumulates allocations per swing
    // nor grows the retained bytes from frame to frame.
    for word in ["gamma", "delta", "theta", "alpha", "zeta", "kappa", "gamma", "delta"] {
        ui.set_middle_word(word.into());
        let (swing_allocs, _swing_net) = do_frame(&mut buffer);
        assert!(swing_allocs <= 2048, "mono swing {word} allocated {swing_allocs} times");
    }

    // And an unchanged frame after the changes stays bounded.
    let (_, third_net) = do_frame(&mut buffer);
    assert!(third_net.abs() <= 1024 * 1024, "frames after changes retained {third_net} bytes");
}
