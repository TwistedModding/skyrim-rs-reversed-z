//! The mirrored twin of a vertex shader (DXBC, shader model 5): the same program with the position's
//! z replaced by w - z. Every write of the position goes to a temp and the shader ends by writing it
//! out. Where clip z comes from matrix rows, the instructions that compute it are also copied in
//! front of each original into fresh temps, reading `w_row - z_row` for the z row, so far depth
//! keeps its precision while every other use of clip z (fog, the far push) keeps the standard value.
//! Otherwise (screen quads, constant depth, the sky's z = w) the shader ends with z = w - z.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const ADD: u32 = 0;
const DP2: u32 = 15;
const DP3: u32 = 16;
const DP4: u32 = 17;
const MAD: u32 = 50;
const MIN: u32 = 51;
const MAX: u32 = 52;
const CUSTOM_DATA: u32 = 53;
const MOV: u32 = 54;
const MUL: u32 = 56;
const RET: u32 = 62;
const RETC: u32 = 63;
const DCL_OUTPUT_SIV: u32 = 103;
const DCL_TEMPS: u32 = 104;
const DECLARATIONS: core::ops::RangeInclusive<u32> = 0x58..=0x6b;
const NAME_POSITION: u32 = 1;

const TEMP: u32 = 0;
const INPUT: u32 = 1;
const OUTPUT: u32 = 2;
const IMMEDIATE32: u32 = 4;
const IMMEDIATE64: u32 = 5;
const CONSTANT_BUFFER: u32 = 8;

const NEGATE: u32 = 1 | (1 << 6);
const EXTENDED: u32 = 1 << 31;
const XYZW: [u8; 4] = [0, 1, 2, 3];
const CB_XYZW: u32 = 2 | (1 << 2) | (0xe4 << 4) | (CONSTANT_BUFFER << 12) | (2 << 20);

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
        if !DECLARATIONS.contains(&op) && op != CUSTOM_DATA {
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
    matches!(op, ADD | MAD | MIN | MAX | MOV | MUL)
}

fn dot_width(op: u32) -> Option<usize> {
    match op {
        DP2 => Some(2),
        DP3 => Some(3),
        DP4 => Some(4),
        _ => None,
    }
}

/// The source components an instruction reads to produce destination component `c`.
fn read_components(op: u32, s: &Operand, c: u8) -> Vec<u8> {
    match dot_width(op) {
        Some(n) => s.swizzle[..n].to_vec(),
        None => vec![s.swizzle[c as usize]],
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
        if !component_wise(ins.op) && dot_width(ins.op).is_none() {
            return None;
        }
        for c in components(hit) {
            need.remove(&(d.kind, r, c));
        }
        members.insert(n, hit);
        rows.extend(ins.operands[1..].iter().filter_map(Operand::element));
        for c in components(hit) {
            for s in &ins.operands[1..] {
                match s.kind {
                    IMMEDIATE32 | IMMEDIATE64 | CONSTANT_BUFFER | INPUT => {}
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

/// What the precise twin adds: instructions before and after given instructions, the temp and
/// component `z` holding mirrored clip z, and the temps used.
struct Mirrored {
    before: HashMap<usize, Vec<u32>>,
    after: HashMap<usize, Vec<u32>>,
    z: (u32, u8),
    extra_temps: u32,
}

/// Temps past the shader's own: the redirected position, `w_row - z_row`, standard clip z and the
/// correction, then the copies'.
const POSITION: u32 = 0;
const ROWS: u32 = 1;
const KEPT: u32 = 2;
const FIRST_COPY: u32 = 3;

/// Fresh temps for the copies, one per original register.
struct Fresh {
    first: u32,
    map: HashMap<(u32, u32), u32>,
}

impl Fresh {
    #[allow(clippy::cast_possible_truncation)]
    fn get(&mut self, key: (u32, u32)) -> u32 {
        let next = self.first + self.map.len() as u32;
        *self.map.entry(key).or_insert(next)
    }
}

/// The instructions of the z slice that depend on the z row, up to the last dot product among them
/// (clip z), and that dot product.
fn clip_z(
    code: &[Instruction],
    z_slice: &BTreeMap<usize, u8>,
    z_row: (u32, u32),
) -> Option<(BTreeSet<usize>, usize)> {
    let mut writer: HashMap<(u32, u8), usize> = HashMap::new();
    let mut dependent: BTreeSet<usize> = BTreeSet::new();
    let mut leaf = None;
    for (n, ins) in code.iter().enumerate() {
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
                dependent.insert(n);
                if dot_width(ins.op).is_some() && hit.is_power_of_two() {
                    leaf = Some(n);
                }
            }
        }
        record_writes(&mut writer, ins, n);
    }
    let leaf = leaf?;
    Some((dependent.range(..=leaf).copied().collect(), leaf))
}

/// The copy of instruction `ins` writing component mask `hit` to a fresh temp, reading `rows` for
/// the z row and the fresh temps for values the other copies wrote.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn copy_instruction(
    tok: &[u32],
    ins: &Instruction,
    hit: u8,
    z_row: (u32, u32),
    rows: u32,
    copied: &BTreeSet<usize>,
    writer: &HashMap<(u32, u8), usize>,
    fresh: &mut Fresh,
) -> Option<Vec<u32>> {
    let d = &ins.operands[0];
    if d.extended {
        return None;
    }
    let mut body = vec![tok[ins.at] & !(0x7f << 24)];
    body.extend_from_slice(&tok[ins.at + 1..d.start]);
    body.push((tok[d.start] & !(0xff << 4) & !(0xff << 12)) | (u32::from(hit) << 4) | (TEMP << 12));
    body.push(fresh.get((d.kind, d.register()?)));
    for s in &ins.operands[1..] {
        let mut t = tok[s.start..s.end].to_vec();
        if s.element() == Some(z_row) {
            let mut head = vec![(t[0] & (((1 << 12) - 1) | EXTENDED)) | (TEMP << 12) | (1 << 20)];
            if s.extended {
                head.push(t[1]);
            }
            head.push(rows);
            t = head;
        } else if s.kind == TEMP {
            let r = s.register()?;
            let inside: HashSet<bool> = components(hit)
                .flat_map(|c| read_components(ins.op, s, c))
                .map(|sc| writer.get(&(r, sc)).is_some_and(|w| copied.contains(w)))
                .collect();
            match inside.into_iter().collect::<Vec<_>>().as_slice() {
                [true] => *t.last_mut()? = fresh.get((TEMP, r)),
                [false] => {}
                _ => return None,
            }
        }
        body.extend(t);
    }
    body[0] |= (body.len() as u32) << 24;
    Some(body)
}

/// Copies, in front of each original, of the instructions that compute clip z (the last dot
/// product that depends on the z row) with `w_row - z_row` read for the z row, writing fresh temps;
/// and a copy of standard clip z kept after the original. The shader then ends with
/// z = mirrored + (standard - final): exact where final is clip z itself, and the far push the
/// engine adds after it carried over.
fn mirrored_slice(
    tok: &[u32],
    code: &[Instruction],
    position: u32,
    base: u32,
    first: usize,
) -> Option<Mirrored> {
    let (z_slice, z_rows) = slice(code, position, 2)?;
    let (_, w_rows) = slice(code, position, 3)?;
    let mut z_only = z_rows.difference(&w_rows);
    let mut w_only = w_rows.difference(&z_rows);
    let (Some(&z_row), None, Some(&w_row), None) =
        (z_only.next(), z_only.next(), w_only.next(), w_only.next())
    else {
        return None;
    };
    if z_row.0 != w_row.0 {
        return None;
    }
    let (copied, leaf) = clip_z(code, &z_slice, z_row)?;

    let mut fresh = Fresh {
        first: base + FIRST_COPY,
        map: HashMap::new(),
    };
    let mut before: HashMap<usize, Vec<u32>> = HashMap::new();
    before.insert(
        first,
        instruction(
            ADD,
            &[
                &reg(TEMP, base + ROWS, Some(0xf), XYZW, false),
                &[CB_XYZW, w_row.0, w_row.1],
                &[CB_XYZW | EXTENDED, NEGATE, z_row.0, z_row.1],
            ],
        ),
    );
    let mut writer: HashMap<(u32, u8), usize> = HashMap::new();
    for (n, ins) in code.iter().enumerate() {
        if copied.contains(&n) {
            let copy = copy_instruction(
                tok,
                ins,
                z_slice[&n],
                z_row,
                base + ROWS,
                &copied,
                &writer,
                &mut fresh,
            )?;
            before.entry(n).or_default().extend(copy);
        }
        record_writes(&mut writer, ins, n);
    }
    let d = &code[leaf].operands[0];
    let c = components(z_slice[&leaf]).next()?;
    let standard = if d.kind == OUTPUT {
        base + POSITION
    } else {
        d.register()?
    };
    let after = HashMap::from([(
        leaf,
        instruction(
            MOV,
            &[
                &reg(TEMP, base + KEPT, Some(0b0001), XYZW, false),
                &reg(TEMP, standard, None, [c; 4], false),
            ],
        ),
    )]);
    let z = (fresh.get((d.kind, d.register()?)), c);
    #[allow(clippy::cast_possible_truncation)]
    let extra_temps = FIRST_COPY + fresh.map.len() as u32;
    Some(Mirrored {
        before,
        after,
        z,
        extra_temps,
    })
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
    let grown = tok.len() * 4 - old * 4;
    let mut twin = container[..at].to_vec();
    twin.extend(shex);
    twin.extend(&container[at + 8 + old * 4..]);
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
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn mirrored(container: &[u8]) -> Option<Vec<u8>> {
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
        .position(|i| !DECLARATIONS.contains(&i.op) && i.op != CUSTOM_DATA)?;
    let base = temps.unwrap_or(0);

    let position_out = |mask: u8| reg(OUTPUT, position, Some(mask), XYZW, false);
    let temp = |r: u32, c: u8, negated: bool| reg(TEMP, base + r, None, [c; 4], negated);
    let mut tail = instruction(
        MOV,
        &[
            &position_out(0b1011),
            &reg(TEMP, base + POSITION, None, XYZW, false),
        ],
    );
    let (before, after, extra) = if let Some(m) = mirrored_slice(&tok, &code, position, base, first)
    {
        tail.extend(instruction(
            ADD,
            &[
                &reg(TEMP, base + KEPT, Some(0b0010), XYZW, false),
                &temp(KEPT, 0, false),
                &temp(POSITION, 2, true),
            ],
        ));
        tail.extend(instruction(
            ADD,
            &[
                &position_out(0b0100),
                &reg(TEMP, m.z.0, None, [m.z.1; 4], false),
                &temp(KEPT, 1, false),
            ],
        ));
        (m.before, m.after, m.extra_temps)
    } else {
        tail.extend(instruction(
            ADD,
            &[
                &position_out(0b0100),
                &temp(POSITION, 3, false),
                &temp(POSITION, 2, true),
            ],
        ));
        (HashMap::new(), HashMap::new(), 1)
    };

    let mut out = tok[..2].to_vec();
    for (n, ins) in code.iter().enumerate() {
        if temps.is_none() && n == first {
            out.extend([DCL_TEMPS | (2 << 24), extra]);
        }
        let mut body = tok[ins.at..ins.at + ins.len].to_vec();
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
        if let Some(b) = before.get(&n) {
            out.extend(b);
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
    Some(rebuild(container, &offsets, at, tok.len(), &out))
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
mod tests {
    use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DDisassemble};
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    /// Clip position from rows of a constant buffer, fog from its length, and the far push.
    const ROWS: &str = "cbuffer view : register(b2) { float4 rows[4]; }\n\
        float4 main(float4 p : POSITION, out float fog : FOG) : SV_Position {\n\
            float4 o = float4(dot(rows[0], p), dot(rows[1], p), dot(rows[2], p), dot(rows[3], p));\n\
            fog = length(o.xyz);\n\
            o.z += 0.5 * saturate((o.z - 70000.0) * 0.0001);\n\
            return o;\n\
        }\n";
    const CONSTANT: &str =
        "float4 main(float4 p : POSITION) : SV_Position { return float4(p.xy, 0.25, 1.0); }\n";

    fn bytes(blob: &ID3DBlob) -> Vec<u8> {
        unsafe {
            core::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        }
        .to_vec()
    }

    fn compile(source: &str) -> Vec<u8> {
        let mut code = None;
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                s!("main"),
                s!("vs_5_0"),
                0,
                0,
                &raw mut code,
                None,
            )
        }
        .unwrap();
        bytes(&code.unwrap())
    }

    fn text(container: &[u8]) -> String {
        let blob =
            unsafe { D3DDisassemble(container.as_ptr().cast(), container.len(), 0, None) }.unwrap();
        String::from_utf8_lossy(&bytes(&blob)).into_owned()
    }

    #[test]
    fn checksum_matches_the_compiler() {
        let code = compile(ROWS);
        assert_eq!(super::checksum(&code), code[4..20]);
    }

    #[test]
    fn clip_z_from_the_rows() {
        let twin = text(&super::mirrored(&compile(ROWS)).unwrap());
        assert!(twin.contains("cb2[3].xyzw, -cb2[2].xyzw"), "{twin}");
        assert!(
            twin.contains("dp3 "),
            "the fog keeps standard clip z: {twin}"
        );
        assert!(twin.contains("add o0.z, r"), "{twin}");
    }

    #[test]
    fn constant_depth_flipped() {
        let twin = text(&super::mirrored(&compile(CONSTANT)).unwrap());
        assert!(twin.contains("dcl_temps 1"), "{twin}");
        assert!(twin.contains("add o0.z, r0.wwww, -r0.zzzz"), "{twin}");
    }
}
