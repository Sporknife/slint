// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

//! Cache of shaped text and broken lines for the built-in text layout.
//!
//! Redrawing an element whose text did not change still pays for shaping and
//! line breaking unless the results are kept: a color change, an overlay
//! moving above the text, or a binding re-evaluating to the same string all
//! redraw the same paragraph. This cache holds the [`ShapeBuffer`] and the
//! broken [`TextLine`]s of the paragraphs laid out most recently, so a call
//! with the same inputs serves them instead of re-running either pass.
//!
//! Entries are keyed by the item that drew them (`None` for text drawn
//! without an item, such as `draw_string`), plus every input that changes
//! the layout result: the string bytes, the font identity, the paragraph
//! geometry, and the alignment, wrap, overflow, line-limit and single-line
//! options.
//! The brush color is recorded per entry but is deliberately not part of
//! the key: it changes nothing about the layout, so a color change keeps
//! the entry servable, while the recorded value lets the dirty-region
//! computation tell a text change from a color change.
//!
//! The number of slots is fixed so the cache works on `no_std` targets and
//! stays bounded in memory; a paragraph whose key no longer matches simply
//! replaces an entry.

use alloc::vec::Vec;
use core::cell::RefCell;

use super::PhysicalLength;
use i_slint_core::Brush;
use i_slint_core::items::{TextHorizontalAlignment, TextOverflow, TextVerticalAlignment, TextWrap};
use i_slint_core::textlayout::{AbstractFont, ShapeBuffer, TextLine, TextParagraphLayout};

/// Number of paragraphs kept per cache instance.
///
/// A frame lays out each dirty text item once, and the dirty-region
/// computation may lay out an item before it is drawn. The slot count must
/// cover every text item a *cold* frame draws: on the first frame (or after
/// a full repaint) every item wants a slot, and with fewer slots than items
/// the round-robin evicts entries of items drawn earlier in the same frame,
/// which disables the narrowing for them until they draw again —
/// permanently, while the scene keeps redrawing more items than slots.
/// Twenty-four slots cover typical screens with margin (the bench scene
/// draws eighteen text paragraphs on a cold frame). More text items than
/// slots still fall back to re-layout, which is correct.
const SLOT_COUNT: usize = 24;

/// The identity of a slot: everything a layout lookup must agree on for the
/// stored shape buffer and lines to be reusable. Lengths are compared
/// exactly; there is no hash, so a hit is always byte-identical to
/// recomputing.
#[derive(PartialEq)]
struct CacheKey {
    /// Bytes of the paragraph text.
    string: Vec<u8>,
    /// Font identity from [`FontShapingIdentity`]: face, pixel size and
    /// variation coordinates.
    font: Vec<u8>,
    letter_spacing: Option<PhysicalLength>,
    line_height: Option<PhysicalLength>,
    max_width: PhysicalLength,
    max_height: PhysicalLength,
    horizontal_alignment: TextHorizontalAlignment,
    vertical_alignment: TextVerticalAlignment,
    wrap: TextWrap,
    overflow: TextOverflow,
    max_lines: Option<usize>,
    single_line: bool,
}

impl CacheKey {
    fn new<Font>(font_identity: &[u8], paragraph: &TextParagraphLayout<'_, Font>) -> Self
    where
        Font: AbstractFont<Length = PhysicalLength>,
    {
        Self {
            string: paragraph.string.as_bytes().to_vec(),
            font: font_identity.to_vec(),
            letter_spacing: paragraph.layout.letter_spacing,
            line_height: paragraph.layout.line_height,
            max_width: paragraph.max_width,
            max_height: paragraph.max_height,
            horizontal_alignment: paragraph.horizontal_alignment,
            vertical_alignment: paragraph.vertical_alignment,
            wrap: paragraph.wrap,
            overflow: paragraph.overflow,
            max_lines: paragraph.max_lines,
            single_line: paragraph.single_line,
        }
    }
}

struct CacheEntry {
    /// The item the layout was computed for. Entries drawn without an item
    /// (from `draw_string`) use `None` and are never consulted by the
    /// dirty-region diff, which needs to know which pixels an item showed
    /// last.
    item: Option<(usize, u32)>,
    key: CacheKey,
    /// The brush the paragraph was drawn with, recorded by the draw path.
    color: Brush,
    shape_buffer: ShapeBuffer<PhysicalLength>,
    lines: Vec<TextLine<PhysicalLength>>,
}

/// Cache of shaped glyphs and broken lines, keyed by item and layout input.
///
/// See the [module documentation](self) for what the cache stores and when
/// entries are replaced.
pub struct TextLayoutCache {
    slots: RefCell<[Option<CacheEntry>; SLOT_COUNT]>,
    /// Next slot to overwrite once every slot holds an entry.
    round_robin: core::cell::Cell<usize>,
}

impl Default for TextLayoutCache {
    fn default() -> Self {
        Self {
            slots: RefCell::new(core::array::from_fn(|_| None)),
            round_robin: core::cell::Cell::new(0),
        }
    }
}

impl TextLayoutCache {
    /// Lays out `paragraph` through [`TextParagraphLayout::layout_broken_lines`],
    /// serving the shaped glyphs and broken lines from the cache when an
    /// entry for the same item, string, font and layout inputs is present,
    /// and computing and storing them otherwise. `run` receives the shape
    /// buffer and the broken lines in both cases.
    ///
    /// `color` is recorded on the entry for the dirty-region diff; it is
    /// deliberately not part of the key (see the [module
    /// documentation](self)). The borrow of the cache entry is held across
    /// `run`, which draws; nothing on that path reads the cache again.
    pub(crate) fn with_layout<Font, R>(
        &self,
        item: Option<(usize, u32)>,
        paragraph: &TextParagraphLayout<'_, Font>,
        font_identity: &[u8],
        color: Brush,
        mut run: impl FnMut(
            &ShapeBuffer<PhysicalLength>,
            &[TextLine<PhysicalLength>],
        ) -> Result<PhysicalLength, R>,
    ) -> Result<PhysicalLength, R>
    where
        Font: AbstractFont<Length = PhysicalLength>,
    {
        let key = CacheKey::new(font_identity, paragraph);

        let hit = {
            let mut slots = self.slots.borrow_mut();
            let entry = slots.iter_mut().find_map(|slot| {
                slot.as_mut().filter(|entry| entry.item == item && entry.key == key)
            });
            entry.map(|entry| {
                entry.color = color.clone();
                run(&entry.shape_buffer, &entry.lines)
            })
        };
        if let Some(result) = hit {
            return result;
        }

        let (shape_buffer, lines) = shape_and_break(paragraph);
        let result = run(&shape_buffer, &lines);
        self.store(item, key, color, shape_buffer, lines);
        result
    }

    fn store(
        &self,
        item: Option<(usize, u32)>,
        key: CacheKey,
        color: Brush,
        shape_buffer: ShapeBuffer<PhysicalLength>,
        lines: Vec<TextLine<PhysicalLength>>,
    ) {
        let mut slots = self.slots.borrow_mut();
        // An item replaces its own previous entry, if any: keeping stale
        // layouts of the same item around wastes slots and lets later
        // lookups find the outdated one first.
        let slot = slots
            .iter()
            .position(|slot| slot.as_ref().is_some_and(|entry| entry.item == item))
            .or_else(|| slots.iter().position(Option::is_none))
            .unwrap_or_else(|| {
                let slot = self.round_robin.get();
                self.round_robin.set((slot + 1) % SLOT_COUNT);
                slot
            });
        slots[slot] = Some(CacheEntry { item, key, color, shape_buffer, lines });
    }

    /// Drops every entry drawn by items of a destroyed tree, so a later tree
    /// reusing the same address can never match its layouts. `tree` is the
    /// address half of the [`item_cache_id`](super::item_cache_id) identity.
    pub(crate) fn evict_tree(&self, tree: usize) {
        let mut slots = self.slots.borrow_mut();
        for slot in slots.iter_mut() {
            let destroyed = slot
                .as_ref()
                .is_some_and(|entry| entry.item.is_some_and(|(item_tree, _)| item_tree == tree));
            if destroyed {
                *slot = None;
            }
        }
    }
}

/// Shapes the paragraph's string and breaks it into lines: the two passes
/// the draw consumes, in one place for the cache's miss path.
fn shape_and_break<Font>(
    paragraph: &TextParagraphLayout<'_, Font>,
) -> (ShapeBuffer<PhysicalLength>, Vec<TextLine<PhysicalLength>>)
where
    Font: AbstractFont<Length = PhysicalLength>,
{
    let shape_buffer = ShapeBuffer::new(&paragraph.layout, paragraph.string);
    let lines = paragraph.break_lines(&shape_buffer);
    (shape_buffer, lines)
}
