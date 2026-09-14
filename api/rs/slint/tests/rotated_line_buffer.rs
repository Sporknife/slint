// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Regression test: line-buffered rendering on a rotated screen must produce
//! the same pixels as buffered rendering.
//!
//! Combining `render_by_line` with `set_rendering_rotation(Rotate90)` tripped
//! a `debug_assert` in the renderer's line walker (`scene.current_line` runs
//! past a span) on a plain full repaint — caught while testing rotated text
//! dirty-region narrowing, which needed this combination.

use slint::platform::software_renderer::{
    LineBufferProvider, MinimalSoftwareWindow, RenderingRotation, RepaintBufferType, Rgb565Pixel,
};
use slint::platform::{PlatformError, WindowAdapter};
use std::rc::Rc;

const WIDTH: u32 = 120;
const HEIGHT: u32 = 80;

thread_local! {
    static QUEUED_WINDOWS: std::cell::RefCell<Vec<Rc<MinimalSoftwareWindow>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

struct TestPlatform;
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

fn queue_window(window: Rc<MinimalSoftwareWindow>) {
    let _ = slint::platform::set_platform(Box::new(TestPlatform));
    QUEUED_WINDOWS.with(|queue| queue.borrow_mut().push(window));
}

/// Row-major view over the frame buffer, transposed for 90° rotation: the
/// rotated frame is HEIGHT pixels wide, so each rendered line is addressed
/// as `line * HEIGHT + range`.
struct RotatedLineBuffer<'a>(&'a mut [Rgb565Pixel]);

impl LineBufferProvider for RotatedLineBuffer<'_> {
    type TargetPixel = Rgb565Pixel;
    fn process_line(
        &mut self,
        line: usize,
        range: core::ops::Range<usize>,
        render_fn: impl FnOnce(&mut [Self::TargetPixel]),
    ) {
        render_fn(&mut self.0[line * HEIGHT as usize..][range]);
    }
}

#[test]
fn rotated_render_by_line_matches_buffered_render() {
    let narrow = MinimalSoftwareWindow::new(RepaintBufferType::SwappedBuffers);
    queue_window(narrow.clone());
    slint::slint! {
        export component TestCase inherits Window {
            background: white;
            Text {
                x: 10px; y: 10px;
                text: "Hello 123";
                color: black;
                font-size: 14px;
            }
        }
    }
    let ui_narrow = TestCase::new().unwrap();
    ui_narrow.show().unwrap();
    let reference = MinimalSoftwareWindow::new(RepaintBufferType::NewBuffer);
    queue_window(reference.clone());
    let ui_reference = TestCase::new().unwrap();
    ui_reference.show().unwrap();

    let size = slint::PhysicalSize::new(WIDTH, HEIGHT);
    narrow.set_size(size);
    reference.set_size(size);

    let mut by_line = vec![Rgb565Pixel::default(); (WIDTH * HEIGHT) as usize];
    narrow.request_redraw();
    assert!(
        narrow.draw_if_needed(|renderer| {
            renderer.set_rendering_rotation(RenderingRotation::Rotate90);
            renderer.render_by_line(RotatedLineBuffer(&mut by_line));
        }),
        "line-buffered rotated frame drew nothing"
    );

    let mut buffered = vec![Rgb565Pixel::default(); (WIDTH * HEIGHT) as usize];
    reference.request_redraw();
    assert!(
        reference.draw_if_needed(|renderer| {
            renderer.set_rendering_rotation(RenderingRotation::Rotate90);
            renderer.render(&mut buffered, HEIGHT as usize);
        }),
        "buffered rotated frame drew nothing"
    );

    assert_eq!(
        by_line, buffered,
        "rotated line-buffered render diverged from rotated buffered render"
    );
}

/// A small non-text change must also work line-buffered under rotation: the
/// dirty region is a sub-rectangle then, and no text hook runs at all, so a
/// failure here implicates the line walker itself rather than any text code.
#[test]
fn rotated_render_by_line_small_rect_no_text() {
    let narrow = MinimalSoftwareWindow::new(RepaintBufferType::SwappedBuffers);
    queue_window(narrow.clone());
    slint::slint! {
        export component RectCase inherits Window {
            background: white;
            in-out property <color> box-color: red;
            Rectangle {
                x: 10px; y: 10px; width: 30px; height: 20px;
                background: box-color;
            }
        }
    }
    let ui = RectCase::new().unwrap();
    ui.show().unwrap();

    let size = slint::PhysicalSize::new(WIDTH, HEIGHT);
    narrow.set_size(size);
    let mut buffer = vec![Rgb565Pixel::default(); (WIDTH * HEIGHT) as usize];
    let draw_lb = |buffer: &mut [Rgb565Pixel]| {
        narrow.request_redraw();
        narrow.draw_if_needed(|renderer| {
            renderer.set_rendering_rotation(RenderingRotation::Rotate90);
            renderer.render_by_line(RotatedLineBuffer(buffer));
        })
    };
    assert!(draw_lb(&mut buffer));
    ui.set_box_color(slint::Color::from_rgb_u8(0, 0, 255));
    assert!(draw_lb(&mut buffer), "small rect change drew nothing");
    // A third frame: the rendered region is now small ∪ small (no full
    // screen anywhere in the union), the shape that broke the walker.
    ui.set_box_color(slint::Color::from_rgb_u8(0, 255, 0));
    assert!(draw_lb(&mut buffer), "second small rect change drew nothing");
}

/// Twelve touching rows, one toggled twice: the third frame renders
/// small ∪ small under rotation, with span edges coinciding with dirty
/// edges — the walker boundary the bench scene tripped over. No text, no
/// narrowing hook anywhere near this path.
#[test]
fn rotated_render_by_line_stacked_rows() {
    const ROWS_W: u32 = 120;
    const ROWS_H: u32 = 240;
    let narrow = MinimalSoftwareWindow::new(RepaintBufferType::SwappedBuffers);
    queue_window(narrow.clone());
    slint::slint! {
        export component RowsCase inherits Window {
            background: white;
            in-out property <color> row-color: red;
            for i in 12: Rectangle {
                x: 0px; y: i * 20px; width: 120px; height: 20px;
                background: i == 5 ? root.row-color : green;
            }
        }
    }
    let ui = RowsCase::new().unwrap();
    ui.show().unwrap();

    narrow.set_size(slint::PhysicalSize::new(ROWS_W, ROWS_H));
    let mut buffer = vec![Rgb565Pixel::default(); (ROWS_W * ROWS_H) as usize];
    /// Same transposed addressing as above, for the 120×240 rows scene.
    struct RowsLineBuffer<'a>(&'a mut [Rgb565Pixel]);
    impl LineBufferProvider for RowsLineBuffer<'_> {
        type TargetPixel = Rgb565Pixel;
        fn process_line(
            &mut self,
            line: usize,
            range: core::ops::Range<usize>,
            render_fn: impl FnOnce(&mut [Self::TargetPixel]),
        ) {
            render_fn(&mut self.0[line * ROWS_H as usize..][range]);
        }
    }
    let draw_lb = |buffer: &mut [Rgb565Pixel]| {
        narrow.request_redraw();
        narrow.draw_if_needed(|renderer| {
            renderer.set_rendering_rotation(RenderingRotation::Rotate90);
            renderer.render_by_line(RowsLineBuffer(buffer));
        })
    };
    assert!(draw_lb(&mut buffer));
    ui.set_row_color(slint::Color::from_rgb_u8(0, 0, 255));
    assert!(draw_lb(&mut buffer), "row recolor drew nothing");
    ui.set_row_color(slint::Color::from_rgb_u8(0, 255, 0));
    assert!(draw_lb(&mut buffer), "second row recolor drew nothing");
}
