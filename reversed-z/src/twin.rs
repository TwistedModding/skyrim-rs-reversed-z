// Copyright (c) 2026 TwistedModding. SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
// distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
//! The mirrored twin of a vertex shader (DXBC, shader model 5): the same program with the position's
//! z replaced by w - z. Every write of the position goes to a temp and the shader ends by writing it
//! out. Where clip z comes from matrix rows, the twin ends with w - z computed from w and the
//! projection's z row instead: A near - (A - 1) w. Clip z itself is
//! a float near the distance, so its rounding alone moves a far surface by about ulp(z) z / near;
//! from w that rounding is scaled by A - 1 (about near / far) and far surfaces keep their order at
//! any near. The far push the engine adds after clip z is carried over, scaled by near / 15.
//! The engine's push is a clamped ramp worked out per vertex, which a large triangle across the
//! ramp interpolates wrongly (distant water against the seabed under it), so the twin gives a
//! push in proportion to clip z in its place: linear, so every triangle interpolates it exactly,
//! and still behind the rest.
//! An unused constant-buffer slot carries the projection and scale. Otherwise (screen quads, constant
//! depth, the sky's z = w) the shader ends with z = w - z.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const ADD: u32 = 0;
const AND: u32 = 1;
const DIV: u32 = 14;
const DP2: u32 = 15;
const DP3: u32 = 16;
const DP4: u32 = 17;
const ENDIF: u32 = 21;
const ENDLOOP: u32 = 22;
const ENDSWITCH: u32 = 23;
const IF: u32 = 31;
const LOOP: u32 = 48;
const LT: u32 = 49;
const MAD: u32 = 50;
const MIN: u32 = 51;
const MAX: u32 = 52;
const CUSTOM_DATA: u32 = 53;
const MOV: u32 = 54;
const MOVC: u32 = 55;
const MUL: u32 = 56;
const RET: u32 = 62;
const RETC: u32 = 63;
const SWITCH: u32 = 76;
const DCL_CONSTANT_BUFFER: u32 = 89;
const DCL_GLOBAL_FLAGS: u32 = 106;
const DCL_OUTPUT_SIV: u32 = 103;
const DCL_INPUT_PS_SGV: u32 = 99;
const DCL_INPUT_PS_SIV: u32 = 100;
const DCL_OUTPUT: u32 = 101;
const DCL_TEMPS: u32 = 104;
/// Declaration opcodes: shader model 4's, shader model 5's and the geometry instance count.
fn declaration(op: u32) -> bool {
    matches!(op, 0x58..=0x6b | 0x8f..=0xa2 | 0xce | CUSTOM_DATA)
}
const NAME_POSITION: u32 = 1;

const TEMP: u32 = 0;
const INPUT: u32 = 1;
const OUTPUT: u32 = 2;
const IMMEDIATE32: u32 = 4;
const IMMEDIATE64: u32 = 5;
const SAMPLER: u32 = 6;
const RESOURCE: u32 = 7;
const CONSTANT_BUFFER: u32 = 8;
const IMMEDIATE_CONSTANT_BUFFER: u32 = 9;
const NULL: u32 = 13;
const OUTPUT_DEPTH: u32 = 12;
const OUTPUT_DEPTH_GREATER_EQUAL: u32 = 38;
const OUTPUT_DEPTH_LESS_EQUAL: u32 = 39;

const NEGATE: u32 = 1 | (1 << 6);
const EXTENDED: u32 = 1 << 31;
const XYZW: [u8; 4] = [0, 1, 2, 3];
const CB_XYZW: u32 = 2 | (1 << 2) | (0xe4 << 4) | (CONSTANT_BUFFER << 12) | (2 << 20);

pub(crate) struct Mirrored {
    pub code: Vec<u8>,
    pub constants: Option<u32>,
}

pub(crate) enum Pixel {
    Unchanged,
    Mirrored(Mirrored),
}

#[derive(Clone)]
struct Operand {
    kind: u32,
    index: Vec<Option<u32>>,
    /// Destination mask (bits x..w) when the operand is in mask mode.
    mask: Option<u8>,
    swizzle: [u8; 4],
    extended: bool,
    start: usize,
    end: usize,
}

impl Operand {
    fn register(&self) -> Option<u32> {
        match self.index.as_slice() {
            [Some(i)] => Some(*i),
            _ => None,
        }
    }

    fn element(&self) -> Option<(u32, u32)> {
        match self.index.as_slice() {
            [Some(slot), Some(element)] if self.kind == CONSTANT_BUFFER => Some((*slot, *element)),
            _ => None,
        }
    }
}

/// The first temp register a twin may use: past the declared count and past every temp the code
/// touches, since a shader rewritten after compiling (ENB's injected prologues) can use registers
/// beyond its `dcl_temps`.
fn temp_base(code: &[Instruction], declared: Option<u32>) -> u32 {
    code.iter()
        .flat_map(|ins| &ins.operands)
        .filter(|o| o.kind == TEMP)
        .filter_map(Operand::register)
        .map(|r| r + 1)
        .chain(declared)
        .max()
        .unwrap_or(0)
}

struct Instruction {
    at: usize,
    len: usize,
    op: u32,
    operands: Vec<Operand>,
}

fn operand(tok: &[u32], mut i: usize) -> Option<(Operand, usize)> {
    let t = *tok.get(i)?;
    let start = i;
    i += 1;
    let (components, selection, bits) = (t & 3, (t >> 2) & 3, (t >> 4) & 0xff);
    let (kind, dims) = ((t >> 12) & 0xff, (t >> 20) & 3);
    if t & EXTENDED != 0 {
        while *tok.get(i)? & EXTENDED != 0 {
            i += 1;
        }
        i += 1;
    }
    let mut index = Vec::new();
    for d in 0..dims {
        match (t >> (22 + 3 * d)) & 7 {
            0 => {
                index.push(Some(*tok.get(i)?));
                i += 1;
            }
            1 => {
                index.push(None);
                i += 2;
            }
            rep @ 2..=4 => {
                index.push(None);
                i += [0, 1, 2][rep as usize - 2];
                i = operand(tok, i)?.1;
            }
            _ => return None,
        }
    }
    if kind == IMMEDIATE32 || kind == IMMEDIATE64 {
        i += if components == 2 { 4 } else { 1 } * if kind == IMMEDIATE64 { 2 } else { 1 };
    }
    let (mask, swizzle) = match (components, selection) {
        (2, 0) => (Some((bits & 0xf) as u8), XYZW),
        (2, 1) => (
            None,
            core::array::from_fn(|k| ((bits >> (2 * k)) & 3) as u8),
        ),
        (2, 2) => (None, [(bits & 3) as u8; 4]),
        _ => (None, [0; 4]),
    };
    Some((
        Operand {
            kind,
            index,
            mask,
            swizzle,
            extended: t & EXTENDED != 0,
            start,
            end: i,
        },
        i,
    ))
}

fn decode(tok: &[u32]) -> Option<Vec<Instruction>> {
    let mut out = Vec::new();
    let mut i = 2;
    while i < tok.len() {
        let op = tok[i] & 0x7ff;
        let len = if op == CUSTOM_DATA {
            *tok.get(i + 1)? as usize
        } else {
            ((tok[i] >> 24) & 0x7f) as usize
        };
        if len == 0 || i + len > tok.len() {
            return None;
        }
        let mut operands = Vec::new();
        if !declaration(op) {
            let mut j = i + 1;
            if tok[i] & EXTENDED != 0 {
                while tok[j] & EXTENDED != 0 {
                    j += 1;
                }
                j += 1;
            }
            while j < i + len {
                let (o, next) = operand(tok, j)?;
                operands.push(o);
                j = next;
            }
        }
        out.push(Instruction {
            at: i,
            len,
            op,
            operands,
        });
        i += len;
    }
    Some(out)
}

fn component_wise(op: u32) -> bool {
    matches!(
        op,
        ADD | AND | DIV | LT | MAD | MIN | MAX | MOV | MOVC | MUL
    )
}

fn dot_width(op: u32) -> Option<usize> {
    match op {
        DP2 => Some(2),
        DP3 => Some(3),
        DP4 => Some(4),
        _ => None,
    }
}

/// The source components an instruction reads to produce destination component `c`; all of them for
/// an instruction not known to work per component.
fn read_components(op: u32, s: &Operand, c: u8) -> Vec<u8> {
    match dot_width(op) {
        Some(n) => s.swizzle[..n].to_vec(),
        None if component_wise(op) => vec![s.swizzle[c as usize]],
        None => s.swizzle.to_vec(),
    }
}

fn components(mask: u8) -> impl Iterator<Item = u8> {
    (0..4).filter(move |c| mask >> c & 1 != 0)
}

fn reg(kind: u32, index: u32, mask: Option<u8>, swizzle: [u8; 4], negated: bool) -> Vec<u32> {
    let mut t = 2 | (kind << 12) | (1 << 20);
    match mask {
        Some(m) => t |= u32::from(m) << 4,
        None => {
            t |= (1 << 2)
                | (swizzle
                    .iter()
                    .enumerate()
                    .map(|(k, &c)| u32::from(c) << (2 * k))
                    .sum::<u32>()
                    << 4);
        }
    }
    if negated {
        vec![t | EXTENDED, NEGATE, index]
    } else {
        vec![t, index]
    }
}

/// Component `c` of element `element` of constant buffer `slot`, replicated.
fn cb(slot: u32, element: u32, c: u8, negated: bool) -> Vec<u32> {
    let t = (CB_XYZW & !(0xff << 4)) | ((u32::from(c) * 0x55) << 4);
    if negated {
        vec![t | EXTENDED, NEGATE, slot, element]
    } else {
        vec![t, slot, element]
    }
}

#[allow(clippy::cast_possible_truncation)]
fn instruction(op: u32, operands: &[&[u32]]) -> Vec<u32> {
    let body: Vec<u32> = operands.concat();
    let mut out = vec![op | ((1 + body.len() as u32) << 24)];
    out.extend(body);
    out
}

/// The slice of the program that computes position component `comp`: per instruction the
/// destination components it contributes, and the constant-buffer elements it reads.
type Slice = (BTreeMap<usize, u8>, BTreeSet<(u32, u32)>);

fn slice(code: &[Instruction], position: u32, comp: u8) -> Option<Slice> {
    let mut need: HashSet<(u32, u32, u8)> = HashSet::from([(OUTPUT, position, comp)]);
    let mut members = BTreeMap::new();
    let mut rows = BTreeSet::new();
    for (n, ins) in code.iter().enumerate().rev() {
        let Some(d) = ins.operands.first() else {
            continue;
        };
        let (Some(mask), Some(r)) = (d.mask, d.register()) else {
            continue;
        };
        if d.kind != TEMP && d.kind != OUTPUT {
            continue;
        }
        let hit: u8 = components(mask)
            .filter(|&c| need.contains(&(d.kind, r, c)))
            .fold(0, |m, c| m | 1 << c);
        if hit == 0 {
            continue;
        }
        for c in components(hit) {
            need.remove(&(d.kind, r, c));
        }
        members.insert(n, hit);
        rows.extend(ins.operands[1..].iter().filter_map(Operand::element));
        for c in components(hit) {
            for s in &ins.operands[1..] {
                match s.kind {
                    IMMEDIATE32 | IMMEDIATE64 | CONSTANT_BUFFER | INPUT | SAMPLER | RESOURCE
                    | IMMEDIATE_CONSTANT_BUFFER | NULL => {}
                    TEMP => {
                        let r = s.register()?;
                        need.extend(
                            read_components(ins.op, s, c)
                                .into_iter()
                                .map(|sc| (TEMP, r, sc)),
                        );
                    }
                    _ => return None,
                }
            }
        }
    }
    need.is_empty().then_some((members, rows))
}

/// Temps past the shader's own: the redirected position, and standard clip z with the twin's work.
const POSITION: u32 = 0;
const KEPT: u32 = 1;
const PUSH: u32 = 2;

/// The last dot product of position z that depends on the z row of a matrix whose w row position w
/// reads (standard clip z) and the component it writes. None where clip z is computed, or the
/// shader returns, inside a branch or a loop: the twin's ending needs the kept z on every path.
struct ClipZ {
    leaf: usize,
    component: u8,
    /// z reads the w row too (a z held at w, where the w row is the only row w reads).
    held: bool,
    /// The instruction that adds the far push to standard z, when only moves and minimums follow
    /// it on the way to the position.
    push: Option<usize>,
}

/// The engine's push at the end of its ramp over the clip z there: the push per unit of clip z
/// that meets the engine's at 80,000.
const LINEAR_PUSH: f32 = 0.5 / 80_000.0;

/// The operand of an `add` or `mad` that is added to `(register, component)`, read unmodified.
fn addend_to(ins: &Instruction, hit: u8, register: u32, component: u8) -> Option<usize> {
    let is_z = |s: &Operand| {
        s.kind == TEMP
            && !s.extended
            && s.register() == Some(register)
            && s.swizzle[hit.trailing_zeros() as usize] == component
    };
    match (ins.op, ins.operands.as_slice()) {
        (MAD, [_, _, _, z]) if is_z(z) => Some(1),
        (ADD, [_, a, z]) if is_z(z) && !is_z(a) => Some(1),
        (ADD, [_, z, a]) if is_z(z) && !is_z(a) => Some(2),
        _ => None,
    }
}

fn clip_z(code: &[Instruction], position: u32) -> Option<ClipZ> {
    let (z_slice, z_rows) = slice(code, position, 2)?;
    let (_, w_rows) = slice(code, position, 3)?;
    let mut z_only = z_rows.difference(&w_rows);
    let (Some(&z_row), None) = (z_only.next(), z_only.next()) else {
        return None;
    };
    let mut w_only = w_rows.difference(&z_rows);
    let (w_row, held) = match (w_only.next(), w_only.next()) {
        (Some(&row), None) => (row, false),
        (None, _) if w_rows.len() == 1 => (*w_rows.first()?, true),
        _ => return None,
    };
    if z_row.0 != w_row.0 {
        return None;
    }
    let mut writer: HashMap<(u32, u8), usize> = HashMap::new();
    let mut dependent: BTreeSet<usize> = BTreeSet::new();
    let mut leaf: Option<ClipZ> = None;
    let mut depth = 0u32;
    for (n, ins) in code.iter().enumerate() {
        match ins.op {
            IF | LOOP | SWITCH => depth += 1,
            ENDIF | ENDLOOP | ENDSWITCH => depth = depth.checked_sub(1)?,
            RET if depth != 0 => return None,
            _ => {}
        }
        if let Some(&hit) = z_slice.get(&n) {
            let reads_row = ins.operands[1..].iter().any(|s| {
                s.element() == Some(z_row)
                    || (s.kind == TEMP
                        && s.register().is_some_and(|r| {
                            components(hit)
                                .flat_map(|c| read_components(ins.op, s, c))
                                .any(|sc| {
                                    writer.get(&(r, sc)).is_some_and(|w| dependent.contains(w))
                                })
                        }))
            });
            if reads_row {
                if depth != 0 {
                    return None;
                }
                dependent.insert(n);
                if dot_width(ins.op).is_some() && hit.is_power_of_two() {
                    leaf = Some(ClipZ {
                        leaf: n,
                        component: components(hit).next()?,
                        held,
                        push: None,
                    });
                } else if let Some(z) = &mut leaf {
                    let d = &code[z.leaf].operands[0];
                    let addend = d
                        .register()
                        .filter(|r| {
                            hit.is_power_of_two()
                                && d.kind == TEMP
                                && writer.get(&(*r, z.component)) == Some(&z.leaf)
                        })
                        .and_then(|r| addend_to(ins, hit, r, z.component));
                    if addend.is_some() {
                        z.push = Some(n);
                    } else if !matches!(ins.op, MOV | MIN) {
                        z.push = None;
                    }
                }
            }
        }
        record_writes(&mut writer, ins, n);
    }
    leaf
}

/// Instructions inserted after given instructions of the program.
type Inserted = HashMap<usize, Vec<u32>>;

/// The sum in `KEPT.z` written to the position, not below w less the shader's own z where the
/// shader held that z under standard z plus the push (by more than rounding): a z held at or just
/// inside the far plane stays there instead of being clipped past it.
fn held_ending(base: u32, position_z: Vec<u32>, push: u8) -> [(u32, Vec<Vec<u32>>); 7] {
    let temp = |r: u32, c: u8, negated: bool| reg(TEMP, base + r, None, [c; 4], negated);
    let kept = |c: u8| reg(TEMP, base + KEPT, Some(1 << c), XYZW, false);
    let w_less_z = || vec![temp(POSITION, 3, false), temp(POSITION, 2, true)];
    let rounding = vec![1 | (IMMEDIATE32 << 12), (1.0f32 / 1_048_576.0).to_bits()];
    [
        (ADD, vec![kept(3), temp(KEPT, 0, false), temp(PUSH, push, false)]),
        (ADD, vec![kept(3), temp(KEPT, 3, false), temp(POSITION, 2, true)]),
        (MUL, vec![kept(1), temp(POSITION, 3, false), rounding]),
        (LT, vec![kept(3), temp(KEPT, 1, false), temp(KEPT, 3, false)]),
        (ADD, [vec![kept(1)], w_less_z()].concat()),
        (MAX, vec![kept(1), temp(KEPT, 1, false), temp(KEPT, 2, false)]),
        (
            MOVC,
            vec![
                position_z,
                temp(KEPT, 3, false),
                temp(KEPT, 1, false),
                temp(KEPT, 2, false),
            ],
        ),
    ]
}

/// The instruction that keeps in `PUSH` what `ins` adds to standard z, and the component kept.
fn keep_push(
    tok: &[u32],
    ins: &Instruction,
    base: u32,
    standard: u32,
    c: u8,
) -> Option<(Vec<u32>, u8)> {
    let mask = ins.operands[0].mask?;
    let first = addend_to(ins, mask, standard, c)?;
    let to = reg(TEMP, base + PUSH, Some(mask), XYZW, false);
    let source = |k: usize| &tok[ins.operands[k].start..ins.operands[k].end];
    let keep = if ins.op == MAD {
        instruction(MUL, &[&to, source(first), source(first + 1)])
    } else {
        instruction(MOV, &[&to, source(first)])
    };
    Some((keep, u8::try_from(mask.trailing_zeros()).ok()?))
}

/// The precise twin's additions: standard clip z kept in `KEPT.x` after the instruction that computes
/// it, the far push kept in `PUSH` before the instruction that adds it (taken from the sum
/// otherwise, which is rounded to clip z's precision), and the ending: the far push, scaled, z = w - z from w, or from standard z where the
/// projection's w row is (0, 0, 0, 1) (orthographic). A z the shader held at w stays on the far
/// plane instead of being clipped past it.
fn precise(
    tok: &[u32],
    code: &[Instruction],
    position: u32,
    base: u32,
    slot: u32,
    ClipZ {
        leaf,
        component: c,
        held,
        push,
    }: ClipZ,
) -> Option<(Inserted, Vec<u32>)> {
    let d = &code[leaf].operands[0];
    let standard = if d.kind == OUTPUT {
        base + POSITION
    } else {
        d.register()?
    };
    let temp = |r: u32, c: u8, negated: bool| reg(TEMP, base + r, None, [c; 4], negated);
    let kept = |c: u8| reg(TEMP, base + KEPT, Some(1 << c), XYZW, false);
    let mut after = HashMap::from([(
        leaf,
        instruction(MOV, &[&kept(0), &reg(TEMP, standard, None, [c; 4], false)]),
    )]);
    let mut push_component = 0;
    let pushed = match push {
        Some(n) => {
            // only the held ending compares with the engine's own push
            if held {
                let (keep, component) = keep_push(tok, &code[n], base, standard, c)?;
                push_component = component;
                after.entry(n - 1).or_default().extend(keep);
            }
            let per_z = vec![1 | (IMMEDIATE32 << 12), (-LINEAR_PUSH).to_bits()];
            (MUL, vec![kept(1), temp(KEPT, 0, false), per_z])
        }
        None => (
            ADD,
            vec![kept(1), temp(KEPT, 0, false), temp(POSITION, 2, true)],
        ),
    };
    let position_z = reg(OUTPUT, position, Some(0b0100), XYZW, false);
    let sum = |to: Vec<u32>| (ADD, vec![to, temp(KEPT, 2, false), temp(KEPT, 1, false)]);
    let mut steps = vec![
        pushed,
        (
            MUL,
            vec![kept(1), temp(KEPT, 1, false), cb(slot, 0, 0, false)],
        ),
        (
            ADD,
            vec![kept(3), temp(POSITION, 3, false), temp(KEPT, 0, true)],
        ),
        (MOV, vec![kept(2), cb(slot, 0, 1, false)]),
        (
            MAD,
            vec![
                kept(2),
                temp(KEPT, 2, true),
                temp(POSITION, 3, false),
                cb(slot, 0, 2, false),
            ],
        ),
        (
            MOVC,
            vec![
                kept(2),
                cb(slot, 0, 3, false),
                temp(KEPT, 3, false),
                temp(KEPT, 2, false),
            ],
        ),
    ];
    if held {
        steps.push(sum(kept(2)));
        steps.extend(held_ending(base, position_z, push_component));
    } else {
        steps.push(sum(position_z));
    }
    let mut out = Vec::new();
    for (op, operands) in steps {
        let operands: Vec<&[u32]> = operands.iter().map(Vec::as_slice).collect();
        out.extend(instruction(op, &operands));
    }
    Some((after, out))
}

fn record_writes(writer: &mut HashMap<(u32, u8), usize>, ins: &Instruction, n: usize) {
    if let Some(d) = ins.operands.first()
        && d.kind == TEMP
        && let (Some(mask), Some(r)) = (d.mask, d.register())
    {
        for c in components(mask) {
            writer.insert((r, c), n);
        }
    }
}

/// A container's chunk offsets, and its program chunk's offset and tokens.
fn program(container: &[u8]) -> Option<(Vec<usize>, usize, Vec<u32>)> {
    let u32_at = |o: usize| {
        container
            .get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")))
    };
    let chunks = u32_at(28)? as usize;
    let offsets: Vec<usize> = (0..chunks)
        .map(|k| u32_at(32 + 4 * k).map(|o| o as usize))
        .collect::<Option<_>>()?;
    let at = *offsets
        .iter()
        .find(|&&o| matches!(container.get(o..o + 4), Some(b"SHEX" | b"SHDR")))?;
    let size = u32_at(at + 4)? as usize;
    let tok = container
        .get(at + 8..at + 8 + size)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    Some((offsets, at, tok))
}

/// `container` with the program chunk at `at` (of `old` tokens) replaced by `tok`, its offsets,
/// size and checksum updated.
#[allow(clippy::cast_possible_truncation)]
fn rebuild(container: &[u8], offsets: &[usize], at: usize, old: usize, tok: &[u32]) -> Vec<u8> {
    let mut shex = container[at..at + 4].to_vec();
    shex.extend((tok.len() as u32 * 4).to_le_bytes());
    shex.extend(tok.iter().flat_map(|t| t.to_le_bytes()));
    replace_chunk(container, offsets, at, old * 4 + 8, &shex)
}

#[allow(clippy::cast_possible_truncation)]
fn replace_chunk(
    container: &[u8],
    offsets: &[usize],
    at: usize,
    old: usize,
    chunk: &[u8],
) -> Vec<u8> {
    let grown = chunk.len() - old;
    let mut twin = container[..at].to_vec();
    twin.extend(chunk);
    twin.extend(&container[at + old..]);
    let total = twin.len() as u32;
    twin[24..28].copy_from_slice(&total.to_le_bytes());
    for (k, &o) in offsets.iter().enumerate() {
        if o > at {
            twin[32 + 4 * k..36 + 4 * k].copy_from_slice(&((o + grown) as u32).to_le_bytes());
        }
    }
    let sum = checksum(&twin);
    twin[4..20].copy_from_slice(&sum);
    twin
}

/// The mirrored twin of vertex shader container `container`, or `None` when it has no position
/// output or a shape this does not rewrite.
pub(crate) fn mirrored(container: &[u8]) -> Option<Mirrored> {
    rewrite_position(container, true)
}

/// The mirrored twin of domain shader container `container`: z = w - z at the end, never the
/// precise form, whose constants are bound on the vertex stage.
pub(crate) fn mirrored_domain(container: &[u8]) -> Option<Mirrored> {
    rewrite_position(container, false)
}

#[allow(clippy::cast_possible_truncation)]
fn rewrite_position(container: &[u8], precise_allowed: bool) -> Option<Mirrored> {
    let (offsets, at, tok) = program(container)?;
    let code = decode(&tok)?;
    let mut position = None;
    let mut temps = None;
    for ins in &code {
        match ins.op {
            DCL_OUTPUT_SIV if tok[ins.at + ins.len - 1] == NAME_POSITION => {
                position = operand(&tok, ins.at + 1).and_then(|(o, _)| o.register());
            }
            DCL_TEMPS => temps = Some(tok[ins.at + 1]),
            RETC => return None,
            _ => {}
        }
    }
    let position = position?;
    let first = code
        .iter()
        .position(|i| !declaration(i.op))?;
    let base = temp_base(&code, temps);

    let position_out = |mask: u8| reg(OUTPUT, position, Some(mask), XYZW, false);
    let temp = |r: u32, c: u8, negated: bool| reg(TEMP, base + r, None, [c; 4], negated);
    let mut tail = instruction(
        MOV,
        &[
            &position_out(0b1011),
            &reg(TEMP, base + POSITION, None, XYZW, false),
        ],
    );
    let mut constants = None;
    // a held z is told from the pushed one by the kept push
    let found = clip_z(&code, position)
        .filter(|z| precise_allowed && (!z.held || z.push.is_some()));
    let held = found.as_ref().is_some_and(|z| z.held);
    let after = if let Some(leaf) = found {
        let slot = (0..14).rev().find(|&slot| {
            !code
                .iter()
                .any(|ins| ins.op == DCL_CONSTANT_BUFFER && tok.get(ins.at + 2) == Some(&slot))
        })?;
        let (after, ending) = precise(&tok, &code, position, base, slot, leaf)?;
        constants = Some(slot);
        tail.extend(ending);
        after
    } else {
        tail.extend(instruction(
            ADD,
            &[
                &position_out(0b0100),
                &temp(POSITION, 3, false),
                &temp(POSITION, 2, true),
            ],
        ));
        HashMap::new()
    };
    let extra = if after.is_empty() { 1 } else { 2 + u32::from(held) };

    let mut declared = after.is_empty();
    let mut out = tok[..2].to_vec();
    for (n, ins) in code.iter().enumerate() {
        let mut body = tok[ins.at..ins.at + ins.len].to_vec();
        if !declared && ins.op != DCL_GLOBAL_FLAGS && ins.op != DCL_CONSTANT_BUFFER {
            out.extend([DCL_CONSTANT_BUFFER | (4 << 24), CB_XYZW, constants?, 1]);
            declared = true;
        }
        if temps.is_none() && n == first {
            out.extend([DCL_TEMPS | (2 << 24), base + extra]);
        }
        if ins.op == DCL_TEMPS {
            body[1] = base + extra;
        }
        for s in ins
            .operands
            .iter()
            .filter(|s| s.kind == OUTPUT && s.register() == Some(position))
        {
            if s.extended {
                return None;
            }
            let r = s.start - ins.at;
            body[r] = (body[r] & !(0xff << 12)) | (TEMP << 12);
            body[s.end - ins.at - 1] = base + POSITION;
        }
        if ins.op == RET {
            out.extend(&tail);
        }
        out.extend(body);
        if let Some(a) = after.get(&n) {
            out.extend(a);
        }
    }
    out[1] = out.len() as u32;
    Some(Mirrored {
        code: rebuild(container, &offsets, at, tok.len(), &out),
        constants,
    })
}

fn reversed_depth_kind(kind: u32) -> Option<u32> {
    match kind {
        OUTPUT_DEPTH => Some(OUTPUT_DEPTH),
        OUTPUT_DEPTH_GREATER_EQUAL => Some(OUTPUT_DEPTH_LESS_EQUAL),
        OUTPUT_DEPTH_LESS_EQUAL => Some(OUTPUT_DEPTH_GREATER_EQUAL),
        _ => None,
    }
}

fn reverse_depth_signature(container: &[u8]) -> Option<Vec<u8>> {
    let (offsets, _, _) = program(container)?;
    let mut result = container.to_vec();
    let mut changed = false;
    for &at in &offsets {
        let (stride, name_at, semantic_at) = match container.get(at..at + 4)? {
            b"OSGN" => (24, 0, 8),
            b"OSG5" => (28, 4, 12),
            b"OSG1" => (32, 4, 12),
            _ => continue,
        };
        let size = u32::from_le_bytes(container.get(at + 4..at + 8)?.try_into().ok()?) as usize;
        let mut payload = container.get(at + 8..at + 8 + size)?.to_vec();
        let word = |data: &[u8], at: usize| {
            data.get(at..at + 4)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u32::from_le_bytes)
        };
        let count = word(&payload, 0)? as usize;
        let start = word(&payload, 4)? as usize;
        for i in 0..count {
            let record = start + i * stride;
            let name = word(&payload, record + name_at)? as usize;
            let name = payload.get(name..)?;
            let name = name.get(..name.iter().position(|&byte| byte == 0)?)?;
            let (semantic, name) = if name.eq_ignore_ascii_case(b"SV_DepthGreaterEqual") {
                (68u32, b"SV_DepthLessEqual\0".as_slice())
            } else if name.eq_ignore_ascii_case(b"SV_DepthLessEqual") {
                (67u32, b"SV_DepthGreaterEqual\0".as_slice())
            } else {
                continue;
            };
            let new_name = u32::try_from(payload.len()).ok()?;
            payload
                .get_mut(record + name_at..record + name_at + 4)?
                .copy_from_slice(&new_name.to_le_bytes());
            payload
                .get_mut(record + semantic_at..record + semantic_at + 4)?
                .copy_from_slice(&semantic.to_le_bytes());
            payload.extend(name);
            changed = true;
        }
        if changed {
            payload.resize(payload.len().next_multiple_of(4), 0);
            let mut chunk = container[at..at + 4].to_vec();
            chunk.extend(u32::try_from(payload.len()).ok()?.to_le_bytes());
            chunk.extend(payload);
            result = replace_chunk(container, &offsets, at, size + 8, &chunk);
            break;
        }
    }
    changed.then_some(result)
}

/// Preserves conventional pixel inputs and reverses explicit depth outputs.
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub(crate) fn pixel(container: &[u8]) -> Option<Pixel> {
    let (offsets, at, tok) = program(container)?;
    if tok[0] >> 16 != 0 {
        return None;
    }
    let code = decode(&tok)?;
    let mut position = None;
    let mut depth = None;
    let mut temps = None;
    for ins in &code {
        if matches!(ins.op, DCL_INPUT_PS_SGV | DCL_INPUT_PS_SIV)
            && tok[ins.at + ins.len - 1] == NAME_POSITION
        {
            let input = operand(&tok, ins.at + 1)?.0;
            if input.mask? & 4 != 0 {
                position = Some((input.register()?, input.mask?));
            }
        } else if ins.op == DCL_OUTPUT {
            let output = operand(&tok, ins.at + 1)?.0;
            if let Some(kind) = reversed_depth_kind(output.kind) {
                if depth.is_some() {
                    return None;
                }
                depth = Some(kind);
            }
        } else if ins.op == DCL_TEMPS {
            temps = Some(tok[ins.at + 1]);
        }
    }
    if position.is_none() && depth.is_none() {
        return Some(Pixel::Unchanged);
    }
    let base = temp_base(&code, temps);
    let depth_temp = base + u32::from(position.is_some());
    let extra = u32::from(position.is_some()) + u32::from(depth.is_some());
    let one = [1 | (IMMEDIATE32 << 12), 1.0f32.to_bits()];
    let mut prologue = Vec::new();
    if let Some((register, mask)) = position {
        prologue.extend(instruction(
            MOV,
            &[
                &reg(TEMP, base, Some(mask), XYZW, false),
                &reg(INPUT, register, None, XYZW, false),
            ],
        ));
        prologue.extend(instruction(
            ADD,
            &[
                &reg(TEMP, base, Some(4), XYZW, false),
                &one,
                &reg(INPUT, register, None, [2; 4], true),
            ],
        ));
    }
    let tail = depth.map_or_else(Vec::new, |kind| {
        instruction(
            ADD,
            &[
                &[1 | (kind << 12)],
                &one,
                &reg(TEMP, depth_temp, None, [0; 4], true),
            ],
        )
    });
    let first = code
        .iter()
        .position(|ins| !declaration(ins.op))?;
    let mut out = tok[..2].to_vec();
    for (n, ins) in code.iter().enumerate() {
        let mut body = tok[ins.at..ins.at + ins.len].to_vec();
        if n == first {
            if temps.is_none() {
                out.extend([DCL_TEMPS | (2 << 24), base + extra]);
            }
            out.extend(&prologue);
        }
        if ins.op == DCL_TEMPS {
            body[1] = base + extra;
        }
        if ins.op == DCL_OUTPUT {
            let kind = (body[1] >> 12) & 0xff;
            if let Some(kind) = reversed_depth_kind(kind) {
                body[1] = (body[1] & !(0xff << 12)) | (kind << 12);
            }
        }
        for source in ins.operands.iter().rev() {
            let start = source.start - ins.at;
            if source.kind == INPUT && position.is_some_and(|p| source.register() == Some(p.0)) {
                body[start] = (body[start] & !(0xff << 12)) | (TEMP << 12);
                body[source.end - ins.at - 1] = base;
            } else if reversed_depth_kind(source.kind).is_some() {
                if source.extended {
                    return None;
                }
                body.splice(
                    start..source.end - ins.at,
                    reg(TEMP, depth_temp, Some(1), XYZW, false),
                );
            }
        }
        if ins.op != CUSTOM_DATA {
            body[0] = (body[0] & !(0x7f << 24)) | ((body.len() as u32) << 24);
        }
        if matches!(ins.op, RET | RETC) {
            out.extend(&tail);
        }
        out.extend(body);
    }
    out[1] = out.len() as u32;
    let mut rewritten = rebuild(container, &offsets, at, tok.len(), &out);
    if depth.is_some_and(|kind| kind != OUTPUT_DEPTH) {
        rewritten = reverse_depth_signature(&rewritten)?;
    }
    Some(Pixel::Mirrored(Mirrored {
        code: rewritten,
        constants: None,
    }))
}

/// The container checksum: MD5 rounds over everything after the checksum, with the bit length
/// placed first in the last block and `(bits >> 2) | 1` last.
#[allow(clippy::cast_possible_truncation)]
fn checksum(container: &[u8]) -> [u8; 16] {
    let data = &container[20..];
    let bits = (data.len() as u32).wrapping_mul(8);
    let mut state = [0x6745_2301_u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let full = data.len() - data.len() % 64;
    for block in data[..full].as_chunks::<64>().0 {
        md5_block(&mut state, block);
    }
    let tail = &data[full..];
    let mut block = [0u8; 64];
    if tail.len() < 56 {
        block[..4].copy_from_slice(&bits.to_le_bytes());
        block[4..4 + tail.len()].copy_from_slice(tail);
        block[4 + tail.len()] = 0x80;
    } else {
        block[..tail.len()].copy_from_slice(tail);
        block[tail.len()] = 0x80;
        md5_block(&mut state, &block);
        block = [0; 64];
        block[..4].copy_from_slice(&bits.to_le_bytes());
    }
    block[60..].copy_from_slice(&((bits >> 2) | 1).to_le_bytes());
    md5_block(&mut state, &block);
    let mut out = [0u8; 16];
    for (k, s) in state.iter().enumerate() {
        out[4 * k..4 * k + 4].copy_from_slice(&s.to_le_bytes());
    }
    out
}

#[allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn md5_block(state: &mut [u32; 4], block: &[u8; 64]) {
    const SHIFTS: [u32; 16] = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    let m: [u32; 16] = core::array::from_fn(|k| {
        u32::from_le_bytes(block[4 * k..4 * k + 4].try_into().expect("four bytes"))
    });
    let [mut a, mut b, mut c, mut d] = *state;
    for i in 0..64_u32 {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let k = ((f64::from(i) + 1.0).sin().abs() * 4_294_967_296.0) as u32;
        let f = f
            .wrapping_add(a)
            .wrapping_add(k)
            .wrapping_add(m[g as usize]);
        a = d;
        d = c;
        c = b;
        b = b.wrapping_add(f.rotate_left(SHIFTS[((i / 16) * 4 + i % 4) as usize]));
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d]) {
        *s = s.wrapping_add(v);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fmt::Write as _;
    use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DDisassemble};
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    /// Clip position from rows of a constant buffer, fog from its length, and the far push.
    pub(crate) const ROWS: &str = "cbuffer view : register(b2) { float4 rows[4]; }\n\
        float4 main(float4 p : POSITION, out float fog : FOG) : SV_Position {\n\
            float4 o = float4(dot(rows[0], p), dot(rows[1], p), dot(rows[2], p), dot(rows[3], p));\n\
            fog = length(o.xyz);\n\
            o.z += 0.5 * saturate((o.z - 70000.0) * 0.0001);\n\
            return o;\n\
        }\n";
    /// The rows read through a position z adjusted with a comparison and a select first.
    const ADJUSTED: &str = "cbuffer view : register(b2) { float4 rows[4]; float4 band; }\n\
        float4 main(float4 p : POSITION) : SV_Position {\n\
            if (abs(p.x - band.x) < band.y) p.z -= p.w / 1000000000.0 + 230.0;\n\
            float4 o = float4(dot(rows[0], p), dot(rows[1], p), dot(rows[2], p), dot(rows[3], p));\n\
            o.z += 0.5 * saturate((o.z - 70000.0) * 0.0001);\n\
            return o;\n\
        }\n";
    pub(crate) const CONSTANT: &str =
        "float4 main(float4 p : POSITION) : SV_Position { return float4(p.xy, 0.25, 1.0); }\n";

    /// `ROWS` with its z held at `scale` times w, as Community Shaders' water does.
    pub(crate) fn held(scale: &str) -> String {
        ROWS.replace(
            "return o;",
            &format!("o.z = min(o.z, o.w * {scale});
 return o;"),
        )
    }

    fn bytes(blob: &ID3DBlob) -> Vec<u8> {
        unsafe {
            core::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        }
        .to_vec()
    }

    pub(crate) fn compile(source: &str) -> Vec<u8> {
        compile_stage(source, s!("vs_5_0"))
    }

    pub(crate) fn compile_stage(source: &str, target: windows::core::PCSTR) -> Vec<u8> {
        let mut code = None;
        let mut errors = None;
        let result = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                s!("main"),
                target,
                0,
                0,
                &raw mut code,
                Some(&raw mut errors),
            )
        };
        assert!(
            result.is_ok(),
            "{source}\n{}",
            errors.as_ref().map_or_else(
                || format!("{result:?}"),
                |errors| String::from_utf8_lossy(&bytes(errors)).into_owned()
            )
        );
        bytes(&code.unwrap())
    }

    fn text(container: &[u8]) -> String {
        let blob =
            unsafe { D3DDisassemble(container.as_ptr().cast(), container.len(), 0, None) }.unwrap();
        String::from_utf8_lossy(&bytes(&blob)).into_owned()
    }

    #[test]
    fn pixel_prologue_follows_shader_model_5_declarations() {
        let code = compile_stage(
            "StructuredBuffer<float4> lights : register(t35);
             float4 main(float4 p : SV_Position) : SV_Target { return lights[(uint)p.x] * p.z; }",
            s!("ps_5_0"),
        );
        let super::Pixel::Mirrored(twin) = super::pixel(&code).unwrap() else {
            panic!("position z is read")
        };
        let text = text(&twin.code);
        let lines: Vec<&str> = text.lines().filter(|l| !l.starts_with("//")).collect();
        let last_declaration = lines.iter().rposition(|l| l.starts_with("dcl_")).unwrap();
        let prologue = lines.iter().position(|l| l.starts_with("mov r")).unwrap();
        assert!(lines.iter().any(|l| l.starts_with("dcl_resource_structured")), "{lines:?}");
        assert!(prologue > last_declaration, "{lines:?}");
    }

    #[test]
    fn pixel_twin_keeps_the_immediate_constant_buffer() {
        let code = compile_stage(
            "static const float4 table[4] = { float4(1, 2, 3, 4), float4(5, 6, 7, 8), float4(9, 1, 2, 3), float4(4, 5, 6, 7) };
             float4 main(float4 p : SV_Position) : SV_Target { return table[(uint)p.x % 4] * p.z; }",
            s!("ps_5_0"),
        );
        let super::Pixel::Mirrored(twin) = super::pixel(&code).unwrap() else {
            panic!("position z is read")
        };
        let custom = |container: &[u8]| {
            let (_, _, tok) = super::program(container).unwrap();
            super::decode(&tok)
                .unwrap()
                .into_iter()
                .find(|ins| ins.op == super::CUSTOM_DATA)
                .map(|ins| (tok[ins.at], tok[ins.at + 1]))
                .unwrap()
        };
        assert_eq!(custom(&twin.code), custom(&code));
        assert_eq!(super::checksum(&twin.code), twin.code[4..20]);
    }

    #[test]
    fn checksum_matches_the_compiler() {
        let code = compile(ROWS);
        assert_eq!(super::checksum(&code), code[4..20]);
    }

    #[test]
    fn clip_z_from_the_rows() {
        let rewritten = super::mirrored(&compile(ROWS)).unwrap();
        assert_eq!(rewritten.constants, Some(13));
        let twin = text(&rewritten.code);
        assert!(!twin.contains("CB12["), "{twin}");
        assert!(
            twin.contains("cb13[0].yyyy"),
            "A - 1 from private constants: {twin}"
        );
        assert!(twin.contains("cb13[0].zzzz"), "{twin}");
        assert!(twin.contains("movc "), "orthographic falls back: {twin}");
        assert!(
            twin.contains("dp3 "),
            "the fog keeps standard clip z: {twin}"
        );
        assert!(twin.contains("add o0.z, r"), "{twin}");
        assert!(twin.contains("dcl_constantbuffer CB13[1]"), "{twin}");
        assert!(twin.contains("cb13[0].x"), "the push is scaled: {twin}");
    }

    #[test]
    fn adjusted_position_z_from_w() {
        let twin = text(&super::mirrored(&compile(ADJUSTED)).unwrap().code);
        assert!(twin.contains("cb13[0].zzzz"), "{twin}");
    }

    const DOTS: &str =
        "float4(dot(rows[0], p), dot(rows[1], p), dot(rows[2], p), dot(rows[3], p))";

    #[test]
    fn a_position_moved_by_a_texture_read_is_precise() {
        let source = format!(
            "Texture2D<float4> offsets : register(t0);
            cbuffer view : register(b2) {{ float4 rows[4]; }}
            float4 main(float4 p : POSITION) : SV_Position {{
                p.xyz += offsets.Load(int3(0, 0, 0)).xyz;
                return {DOTS};
            }}
"
        );
        assert_eq!(super::mirrored(&compile(&source)).unwrap().constants, Some(13));
    }

    #[test]
    fn clip_z_inside_a_branch_keeps_w_minus_z() {
        let source = format!(
            "cbuffer view : register(b2) {{ float4 rows[4]; float4 band; }}
            float4 main(float4 p : POSITION) : SV_Position {{
                float4 o = p;
                [branch] if (band.x > 0) o = {DOTS};
                return o;
            }}
"
        );
        let rewritten = super::mirrored(&compile(&source)).unwrap();
        let twin = text(&rewritten.code);
        assert!(twin.contains("if_"), "the branch must survive compilation: {twin}");
        assert_eq!(rewritten.constants, None);
    }

    #[test]
    fn constant_depth_flipped() {
        let rewritten = super::mirrored(&compile(CONSTANT)).unwrap();
        assert_eq!(rewritten.constants, None);
        let twin = text(&rewritten.code);
        assert!(twin.contains("dcl_temps 1"), "{twin}");
        assert!(twin.contains("add o0.z, r0.wwww, -r0.zzzz"), "{twin}");
    }

    pub(crate) fn occupied_slots(slots: &[u32]) -> String {
        let mut source = String::new();
        let mut extra = String::new();
        for slot in slots {
            writeln!(
                source,
                "cbuffer extra{slot} : register(b{slot}) {{ float4 value{slot}; }}"
            )
            .unwrap();
            write!(extra, " + value{slot}.x").unwrap();
        }
        source + &ROWS.replace("length(o.xyz)", &format!("length(o.xyz){extra}"))
    }

    #[test]
    fn occupied_constant_slots_are_preserved() {
        let source = occupied_slots(&[12, 13]);
        let rewritten = super::mirrored(&compile(&source)).unwrap();
        assert_eq!(rewritten.constants, Some(11));
        let disassembly = text(&rewritten.code);
        for slot in [11, 12, 13] {
            assert!(
                disassembly.contains(&format!("dcl_constantbuffer CB{slot}[1]")),
                "{disassembly}"
            );
        }
    }

    #[test]
    fn a_precise_shader_without_a_free_slot_is_refused() {
        let slots: Vec<u32> = (0..14).filter(|&slot| slot != 2).collect();
        assert!(super::mirrored(&compile(&occupied_slots(&slots))).is_none());
    }

    #[test]
    fn twin_temps_start_past_registers_used_beyond_the_declaration() {
        let source = "float4 main(float4 p : SV_Position, float4 a : TEXCOORD0, float4 b : TEXCOORD1) : SV_Target {             float4 x = a * p.z; float4 y = b * x; float4 z = sin(y) * a; return x + y + z * p.z; }";
        let mut container = compile_stage(source, s!("ps_5_0"));
        let (_, at, tok) = super::program(&container).unwrap();
        let code = super::decode(&tok).unwrap();
        let used = super::temp_base(&code, None);
        assert!(used >= 2, "the shader should use several temps, used {used}");
        let declaration = code.iter().find(|i| i.op == super::DCL_TEMPS).unwrap();
        let offset = at + (declaration.at + 1) * 4;
        container[offset..offset + 4].copy_from_slice(&1u32.to_le_bytes());
        let super::Pixel::Mirrored(rewritten) = super::pixel(&container).unwrap() else {
            panic!("expected a position rewrite");
        };
        let disassembly = text(&rewritten.code);
        assert!(disassembly.contains(&format!("dcl_temps {}", used + 1)), "{disassembly}");
        assert!(disassembly.contains(&format!("mov r{used}.z, v0.xyzw")), "{disassembly}");
    }

    #[test]
    fn a_domain_shader_twin_writes_w_minus_z_without_constants() {
        let source = "cbuffer view : register(b0) { float4x4 m; }\n\
            struct P { float3 p : POSITION; };\n\
            struct F { float e[3] : SV_TessFactor; float i : SV_InsideTessFactor; };\n\
            [domain(\"tri\")] float4 main(F f, float3 uvw : SV_DomainLocation, const OutputPatch<P, 3> c) : SV_Position {\n\
            float3 w = c[0].p * uvw.x + c[1].p * uvw.y + c[2].p * uvw.z; return mul(m, float4(w, 1)); }";
        let rewritten = super::mirrored_domain(&compile_stage(source, s!("ds_5_0"))).unwrap();
        let disassembly = text(&rewritten.code);
        assert!(disassembly.contains("ds_5_0"), "{disassembly}");
        assert!(!disassembly.contains("cb13") && !disassembly.contains("cb12"), "{disassembly}");
        assert!(disassembly.contains("add o0.z"), "{disassembly}");
    }

    #[test]
    fn conservative_pixel_depth_updates_the_output_signature() {
        for (before, after) in [
            ("SV_DepthGreaterEqual", "SV_DepthLessEqual"),
            ("SV_DepthLessEqual", "SV_DepthGreaterEqual"),
        ] {
            let source = format!(
                "float main(noperspective centroid float4 p : SV_Position) : {before} {{ return p.z; }}"
            );
            let super::Pixel::Mirrored(rewritten) =
                super::pixel(&compile_stage(&source, s!("ps_5_0"))).unwrap()
            else {
                panic!("expected a depth rewrite");
            };
            let disassembly = text(&rewritten.code);
            assert!(disassembly.contains(after), "{disassembly}");
            assert!(!disassembly.contains(before), "{disassembly}");
        }
    }
}
