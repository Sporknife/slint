// Copyright © SixtyFPS GmbH <info@slint.dev>
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Slint-Royalty-free-2.0 OR LicenseRef-Slint-Software-3.0

use std::collections::BTreeSet;

fn main() {
    // The coverage check reads bench.rs, so the build script must re-run
    // whenever it changes (slint-build only watches the .slint inputs).
    println!("cargo:rerun-if-changed=bench.rs");
    assert_glyph_coverage();
    slint_build::compile_with_config(
        "ui/bench.slint",
        slint_build::CompilerConfiguration::new()
            .embed_resources(slint_build::EmbedResourcesKind::EmbedForSoftwareRenderer),
    )
    .unwrap();
}

/// Every non-ASCII character the Rust tests drive must appear in a .slint
/// string literal (see the `coverage` manifest in ui/bench.slint), or
/// `EmbedForSoftwareRenderer` will not embed it and the test will silently
/// exercise tofu (fixed invisible advances) instead of real glyphs.
///
/// This scans the string literals of bench.rs — decoding Rust `\u{...}` and
/// `\xNN` escapes — and fails the build listing any non-ASCII character that
/// no .slint literal provides. ASCII, U+25CF and U+2026 are always embedded
/// and exempt. Comments are not literals and never count on either side.
fn assert_glyph_coverage() {
    let rust = std::fs::read_to_string("bench.rs").expect("bench.rs must be readable");
    let slint = std::fs::read_to_string("ui/bench.slint").expect("ui/bench.slint must be readable");
    let mut needed = BTreeSet::new();
    collect_literal_chars(&rust, true, &mut needed);
    // Intentionally tofu, documented per character: no font vendored in this
    // crate can render them, so the embedder could never provide them. They
    // stay in the tests on purpose — deterministic invisible advances that
    // exercise the missing-glyph and multi-byte-offset paths.
    //   U+0301 combining acute (fuzz "cafe"+mark): real combining-mark
    //     coverage comes from Thai U+0E31 instead.
    //   U+200D zero-width joiner (fuzz "a"+mark+"b"): present in the
    //     Hebrew/Thai sources, but the fuzz drives it under the mono face,
    //     which lacks it.
    //   U+1F60A (fuzz smiley): no embeddable outline source here at all.
    for exempt in ['\u{301}', '\u{200d}', '\u{1f60a}'] {
        needed.remove(&exempt);
    }
    let mut provided = BTreeSet::new();
    collect_literal_chars(&slint, false, &mut provided);
    // Always embedded alongside ASCII, independent of any literal.
    provided.insert('●');
    provided.insert('…');
    let missing: Vec<char> = needed.difference(&provided).copied().collect();
    assert!(
        missing.is_empty(),
        "glyph coverage gap: bench.rs drives these characters, but no \
         ui/bench.slint string literal provides them, so EmbedForSoftwareRenderer \
         would leave them as tofu: {}\n\
         Add them to the `coverage` manifest property in ui/bench.slint.",
        missing.iter().map(|c| format!("U+{:04X} {c}", *c as u32)).collect::<Vec<_>>().join(", "),
    );
}

/// Collects the non-ASCII characters of every `"..."` literal in `source`.
/// `//` starts a line comment only outside a literal (so URLs inside strings
/// are safe); `\` starts an escape whose decoded character is recorded for
/// Rust (`rust_escapes`, covering `\u{...}` and `\xNN`) and skipped for
/// .slint (whose escapes always decode to ASCII). The crate uses no raw
/// strings, byte literals or block comments on either side.
fn collect_literal_chars(source: &str, rust_escapes: bool, out: &mut BTreeSet<char>) {
    let mut chars = source.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '"' {
            if c == '/' && chars.peek() == Some(&'/') {
                while chars.next().is_some_and(|c| c != '\n') {}
            }
            continue;
        }
        while let Some(c) = chars.next() {
            if c == '"' {
                break;
            }
            if c != '\\' {
                push_non_ascii(c, out);
                continue;
            }
            match chars.next() {
                Some('u') if rust_escapes => {
                    assert_eq!(chars.next(), Some('{'), "malformed \\u escape in bench.rs");
                    let hex: String = chars.by_ref().take_while(|&c| c != '}').collect();
                    let c = char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap();
                    push_non_ascii(c, out);
                }
                Some('x') if rust_escapes => {
                    let hex: String = chars.by_ref().take(2).collect();
                    let c = char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap();
                    push_non_ascii(c, out);
                }
                // \n, \t, \", \\, \{, \} and friends all decode to ASCII.
                Some(_) => {}
                None => break,
            }
        }
    }
}

fn push_non_ascii(c: char, out: &mut BTreeSet<char>) {
    if !c.is_ascii() {
        out.insert(c);
    }
}
