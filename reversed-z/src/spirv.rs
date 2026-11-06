// Copyright (c) 2026 TwistedModding. SPDX-License-Identifier: MPL-2.0-no-copyleft-exception
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not
// distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.
//! The mirrored twin of a SPIR-V vertex shader: before every return of the vertex entry point,
//! the position's z is replaced. Precisely, from w and the projection's z row: A near - (A - 1) w,
//! with A near read as `-CameraViewProj[2][3]` from the frame block the shader reaches through its
//! resource heap (the projection's own constant, bit for bit), A - 1 as A near / far, and the LOD
//! land's far push carried over scaled by near / 15. Otherwise, plainly, z = w - z. The position is
//! the output decorated `BuiltIn Position`, either a variable or a member of a block.

use std::collections::{HashMap, HashSet};

const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_ENTRY_POINT: u16 = 15;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_COMPOSITE_INSERT: u16 = 82;
const OP_FNEGATE: u16 = 127;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;
const MODEL_VERTEX: u32 = 0;
const STORAGE_UNIFORM: u32 = 2;
const STORAGE_OUTPUT: u32 = 3;
const BUILTIN: u32 = 11;
const OFFSET: u32 = 35;
const POSITION: u32 = 0;
const MAGIC: u32 = 0x0723_0203;
const GLSL_FCLAMP: u32 = 43;
/// `FrameBuffer::CameraViewProj` sits at byte 128 of the frame block (`packoffset(c8)`).
const VIEW_PROJECTION_OFFSET: u32 = 128;
/// The engine's far push: z += 0.5 saturate((z - 70000) / 10000), carried over by near / 15.
const PUSH_START: f32 = 70_000.0;
const PUSH_SLOPE: f32 = 0.0001;
const PUSH_HEIGHT: f32 = 0.5;
const VANILLA_NEAR: f32 = 15.0;

fn opcode(op: u16, len: u32) -> u32 {
    u32::from(op) | (len << 16)
}

struct Instruction {
    op: u16,
    at: usize,
    len: usize,
}

fn instructions(code: &[u32]) -> Option<Vec<Instruction>> {
    let mut out = Vec::new();
    let mut at = 5;
    while at < code.len() {
        let len = (code[at] >> 16) as usize;
        if len == 0 || at + len > code.len() {
            return None;
        }
        out.push(Instruction {
            op: (code[at] & 0xffff) as u16,
            at,
            len,
        });
        at += len;
    }
    Some(out)
}

/// The vertex shader is one.
pub(crate) fn is_vertex(code: &[u32]) -> bool {
    code.len() > 5
        && code[0] == MAGIC
        && instructions(code).is_some_and(|ins| {
            ins.iter()
                .any(|i| i.op == OP_ENTRY_POINT && code[i.at + 1] == MODEL_VERTEX)
        })
}

fn word(code: &[u32], ins: &Instruction, n: usize) -> Option<u32> {
    (n < ins.len).then(|| code[ins.at + n])
}

/// Where the shader reads the frame block's view-projection: the heap variable, its index (a
/// constant or a value of the entry's first block), the member's index constant and its type.
struct Source {
    variable: u32,
    index: Option<u32>,
    member: u32,
    int: u32,
}

/// What the module declares, gathered in one pass.
#[derive(Default)]
struct Module {
    entry: Option<u32>,
    position_var: Option<u32>,
    position_member: Option<(u32, u32)>,
    float: Option<u32>,
    vector: Option<u32>,
    matrix: Option<u32>,
    int: Option<u32>,
    glsl: Option<u32>,
    pointers: HashMap<u32, (u32, u32)>,
    runtime_arrays: HashMap<u32, u32>,
    structs: HashMap<u32, Vec<u32>>,
    view_projection_blocks: HashMap<u32, u32>,
    constants: HashMap<u32, (u32, u32)>,
    outputs: Vec<(u32, u32)>,
    uniforms: Vec<(u32, u32)>,
    first_function: Option<usize>,
    has_push: bool,
}

fn gather(code: &[u32], ins: &[Instruction]) -> Module {
    let mut m = Module::default();
    for i in ins {
        let w = |n| word(code, i, n);
        match i.op {
            OP_ENTRY_POINT if w(1) == Some(MODEL_VERTEX) && m.entry.is_none() => m.entry = w(2),
            OP_EXT_INST_IMPORT
                if code[i.at + 2..i.at + i.len]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .take_while(|&b| b != 0)
                    .eq(*b"GLSL.std.450") =>
            {
                m.glsl = w(1);
            }
            OP_DECORATE if w(2) == Some(BUILTIN) && w(3) == Some(POSITION) => {
                m.position_var = w(1);
            }
            OP_MEMBER_DECORATE if w(3) == Some(BUILTIN) && w(4) == Some(POSITION) => {
                m.position_member = w(1).zip(w(2));
            }
            OP_MEMBER_DECORATE
                if w(3) == Some(OFFSET) && w(4) == Some(VIEW_PROJECTION_OFFSET) =>
            {
                if let (Some(s), Some(member)) = (w(1), w(2)) {
                    m.view_projection_blocks.insert(s, member);
                }
            }
            OP_TYPE_FLOAT if w(2) == Some(32) => m.float = w(1),
            OP_TYPE_INT if w(2) == Some(32) && m.int.is_none() => m.int = w(1),
            OP_TYPE_VECTOR if w(3) == Some(4) && w(2) == m.float => m.vector = w(1),
            OP_TYPE_MATRIX if w(3) == Some(4) && w(2) == m.vector => m.matrix = w(1),
            OP_TYPE_RUNTIME_ARRAY => {
                if let (Some(id), Some(element)) = (w(1), w(2)) {
                    m.runtime_arrays.insert(id, element);
                }
            }
            OP_TYPE_STRUCT => {
                if let Some(id) = w(1) {
                    m.structs.insert(id, code[i.at + 2..i.at + i.len].to_vec());
                }
            }
            OP_TYPE_POINTER => {
                if let (Some(id), Some(storage), Some(pointee)) = (w(1), w(2), w(3)) {
                    m.pointers.insert(id, (storage, pointee));
                }
            }
            OP_CONSTANT => {
                if let (Some(ty), Some(id), Some(value)) = (w(1), w(2), w(3)) {
                    if Some(ty) == m.float && value == PUSH_START.to_bits() {
                        m.has_push = true;
                    }
                    m.constants.insert(id, (ty, value));
                }
            }
            OP_VARIABLE if w(3) == Some(STORAGE_OUTPUT) => {
                if let (Some(id), Some(ty)) = (w(2), w(1)) {
                    m.outputs.push((id, ty));
                }
            }
            OP_VARIABLE if w(3) == Some(STORAGE_UNIFORM) => {
                if let (Some(id), Some(ty)) = (w(2), w(1)) {
                    m.uniforms.push((id, ty));
                }
            }
            OP_FUNCTION if m.first_function.is_none() => m.first_function = Some(i.at),
            _ => {}
        }
    }
    m
}

/// The ids the entry point's first block defines, which every later instruction may use.
fn first_block(code: &[u32], ins: &[Instruction], entry: u32) -> HashSet<u32> {
    let mut ids = HashSet::new();
    let mut inside = false;
    let mut labels = 0;
    for i in ins {
        match i.op {
            OP_FUNCTION => {
                inside = word(code, i, 2) == Some(entry);
                labels = 0;
            }
            OP_FUNCTION_END => inside = false,
            OP_LABEL if inside => labels += 1,
            _ if inside && labels == 1 && i.len > 2 => {
                ids.insert(code[i.at + 2]);
            }
            _ => {}
        }
    }
    ids
}

fn source(code: &[u32], ins: &[Instruction], m: &Module, usable: &HashSet<u32>) -> Option<Source> {
    let matrix = m.matrix?;
    let blocks: Vec<(u32, u32)> = m
        .view_projection_blocks
        .iter()
        .filter(|(s, member)| {
            m.structs
                .get(s)
                .and_then(|members| members.get(**member as usize))
                == Some(&matrix)
        })
        .map(|(s, member)| (*s, *member))
        .collect();
    for i in ins.iter().filter(|i| i.op == OP_ACCESS_CHAIN) {
        let (Some(variable), Some(first)) = (word(code, i, 3), word(code, i, 4)) else {
            continue;
        };
        let Some((_, pointer)) = m.uniforms.iter().find(|(id, _)| *id == variable) else {
            continue;
        };
        let (_, target) = *m.pointers.get(pointer)?;
        let (block, index) = match m.runtime_arrays.get(&target) {
            Some(element) => (*element, Some(first)),
            None => (target, None),
        };
        let Some((_, member)) = blocks.iter().find(|(s, _)| *s == block) else {
            continue;
        };
        if index.is_some_and(|id| !m.constants.contains_key(&id) && !usable.contains(&id)) {
            continue;
        }
        let member_index = match index {
            Some(_) => match word(code, i, 5) {
                Some(id) => id,
                None => {
                    return Some(Source {
                        variable,
                        index,
                        member: *m
                            .constants
                            .iter()
                            .find(|(_, (ty, v))| Some(*ty) == m.int && *v == *member)
                            .map(|(id, _)| id)?,
                        int: m.int?,
                    });
                }
            },
            None => first,
        };
        let Some((int, value)) = m.constants.get(&member_index) else {
            continue;
        };
        if *value != *member {
            continue;
        }
        return Some(Source {
            variable,
            index,
            member: member_index,
            int: *int,
        });
    }
    None
}

/// Words being written and the next free id.
struct Emitter {
    bound: u32,
    words: Vec<u32>,
}

impl Emitter {
    fn id(&mut self) -> u32 {
        self.bound += 1;
        self.bound - 1
    }

    fn emit(&mut self, words: &[u32]) {
        self.words.extend_from_slice(words);
    }

    fn binary(&mut self, op: u16, ty: u32, a: u32, b: u32) -> u32 {
        let id = self.id();
        self.emit(&[opcode(op, 5), ty, id, a, b]);
        id
    }

    fn constant(&mut self, ty: u32, value: u32) -> u32 {
        let id = self.id();
        self.emit(&[opcode(OP_CONSTANT, 4), ty, id, value]);
        id
    }
}

/// The precise ending's ingredients: the source, the uniform float pointer type, the row and
/// column index constants and the float constants 1 / far, 1 / 15, 70000, 0.0001, 0.5, 0, 1.
struct Precise {
    source: Source,
    pointer: u32,
    row: u32,
    column: u32,
    floats: [u32; 7],
}

/// The twin, or `None` when the shader has no vertex entry point writing a position. With `far`
/// the ending is precise where the frame block is reachable.
#[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
pub(crate) fn mirrored(code: &[u32], far: Option<f32>) -> Option<Vec<u32>> {
    if code.len() < 6 || code[0] != MAGIC {
        return None;
    }
    let ins = instructions(code)?;
    let module = gather(code, &ins);
    let entry = module.entry?;
    let (float, vector) = (module.float?, module.vector?);
    let first_function = module.first_function?;
    let (variable, member) = match (module.position_var, module.position_member) {
        (Some(v), _) if module.outputs.iter().any(|(id, _)| *id == v) => (v, None),
        (_, Some((block, member))) => {
            let (id, _) = *module.outputs.iter().find(|(_, pointer)| {
                module.pointers
                    .get(pointer)
                    .is_some_and(|(_, pointee)| *pointee == block)
            })?;
            (id, Some(member))
        }
        _ => return None,
    };
    let usable = first_block(code, &ins, entry);
    let source = far
        .filter(|far| *far > 0.0)
        .and_then(|far| source(code, &ins, &module, &usable).map(|s| (s, far)));
    let find_pointer = |storage: u32, pointee: u32| {
        module.pointers
            .iter()
            .find(|(_, (s, p))| *s == storage && *p == pointee)
            .map(|(id, _)| *id)
    };
    let existing = |int: u32, value: u32| {
        module.constants
            .iter()
            .find(|(_, (ty, v))| *ty == int && *v == value)
            .map(|(id, _)| *id)
    };

    let mut out = Emitter {
        bound: code[3],
        words: Vec::new(),
    };
    let pointer_out = find_pointer(STORAGE_OUTPUT, vector).unwrap_or_else(|| {
        let id = out.id();
        out.emit(&[opcode(OP_TYPE_POINTER, 4), id, STORAGE_OUTPUT, vector]);
        id
    });
    let index = match member {
        Some(member) => {
            let int = module.int.unwrap_or_else(|| {
                let id = out.id();
                out.emit(&[opcode(OP_TYPE_INT, 4), id, 32, 0]);
                id
            });
            Some(existing(int, member).unwrap_or_else(|| out.constant(int, member)))
        }
        None => None,
    };
    let precise = source.map(|(source, far)| {
        let pointer = find_pointer(STORAGE_UNIFORM, float).unwrap_or_else(|| {
            let id = out.id();
            out.emit(&[opcode(OP_TYPE_POINTER, 4), id, STORAGE_UNIFORM, float]);
            id
        });
        let row = existing(source.int, 2).unwrap_or_else(|| out.constant(source.int, 2));
        let column = existing(source.int, 3).unwrap_or_else(|| out.constant(source.int, 3));
        let floats = [
            1.0 / far,
            1.0 / VANILLA_NEAR,
            PUSH_START,
            PUSH_SLOPE,
            PUSH_HEIGHT,
            0.0,
            1.0,
        ]
        .map(|value| out.constant(float, value.to_bits()));
        Precise {
            source,
            pointer,
            row,
            column,
            floats,
        }
    });

    let globals = core::mem::take(&mut out.words);
    out.words = code[..first_function].to_vec();
    out.emit(&globals);
    let mut inside = false;
    for i in &ins {
        if i.at < first_function {
            continue;
        }
        match i.op {
            OP_FUNCTION => inside = word(code, i, 2) == Some(entry),
            OP_FUNCTION_END => inside = false,
            OP_RETURN if inside => {
                let pointer = match index {
                    Some(index) => {
                        let id = out.id();
                        out.emit(&[opcode(OP_ACCESS_CHAIN, 5), pointer_out, id, variable, index]);
                        id
                    }
                    None => variable,
                };
                let loaded = out.id();
                out.emit(&[opcode(OP_LOAD, 4), vector, loaded, pointer]);
                let z = out.id();
                out.emit(&[opcode(OP_COMPOSITE_EXTRACT, 5), float, z, loaded, 2]);
                let w = out.id();
                out.emit(&[opcode(OP_COMPOSITE_EXTRACT, 5), float, w, loaded, 3]);
                let flipped = match &precise {
                    Some(p) => {
                        let [inv_far, inv_near, start, slope, height, zero, one] = p.floats;
                        let chain = out.id();
                        let mut words = vec![0, p.pointer, chain, p.source.variable];
                        words.extend(p.source.index);
                        words.extend([p.source.member, p.row, p.column]);
                        words[0] = opcode(OP_ACCESS_CHAIN, words.len() as u32);
                        out.emit(&words);
                        let negated = out.id();
                        out.emit(&[opcode(OP_LOAD, 4), float, negated, chain]);
                        let a_near = out.id();
                        out.emit(&[opcode(OP_FNEGATE, 4), float, a_near, negated]);
                        let a_less_one = out.binary(OP_FMUL, float, a_near, inv_far);
                        let scaled_w = out.binary(OP_FMUL, float, a_less_one, w);
                        let base = out.binary(OP_FSUB, float, a_near, scaled_w);
                        match module.glsl {
                            Some(glsl) if module.has_push => {
                                let a_w = out.binary(OP_FADD, float, scaled_w, w);
                                let rows = out.binary(OP_FSUB, float, a_w, a_near);
                                let past = out.binary(OP_FSUB, float, rows, start);
                                let ramp = out.binary(OP_FMUL, float, past, slope);
                                let clamped = out.id();
                                out.emit(&[
                                    opcode(OP_EXT_INST, 8),
                                    float,
                                    clamped,
                                    glsl,
                                    GLSL_FCLAMP,
                                    ramp,
                                    zero,
                                    one,
                                ]);
                                let push = out.binary(OP_FMUL, float, clamped, height);
                                let scale = out.binary(OP_FMUL, float, a_near, inv_near);
                                let carried = out.binary(OP_FMUL, float, push, scale);
                                out.binary(OP_FSUB, float, base, carried)
                            }
                            _ => base,
                        }
                    }
                    None => out.binary(OP_FSUB, float, w, z),
                };
                let inserted = out.id();
                out.emit(&[opcode(OP_COMPOSITE_INSERT, 6), vector, inserted, flipped, loaded, 2]);
                out.emit(&[opcode(OP_STORE, 3), pointer, inserted]);
            }
            _ => {}
        }
        out.emit(&code[i.at..i.at + i.len]);
    }
    out.words[3] = out.bound;
    Some(out.words)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OP_TYPE_VOID: u16 = 19;
    const OP_TYPE_FUNCTION: u16 = 33;
    const OP_CONSTANT_COMPOSITE: u16 = 44;

    /// A vertex shader storing a constant position; with `block` through a `gl_PerVertex`
    /// member, with `frame` reading a frame block from a heap first, with `push` holding the
    /// far push's constant.
    fn module(block: bool, frame: bool, push: bool) -> Vec<u32> {
        let mut c = vec![MAGIC, 0x0001_0000, 0, 40, 0];
        c.extend([opcode(17, 2), 1]);
        c.extend([
            opcode(OP_EXT_INST_IMPORT, 6),
            30,
            0x4c53_4c47,
            0x6474_732e,
            0x3035_342e,
            0,
        ]);
        c.extend([opcode(14, 3), 0, 1]);
        c.extend([opcode(OP_ENTRY_POINT, 5), MODEL_VERTEX, 10, 0x006e_6961, 11]);
        if block {
            c.extend([opcode(OP_MEMBER_DECORATE, 5), 9, 0, BUILTIN, POSITION]);
        } else {
            c.extend([opcode(OP_DECORATE, 4), 11, BUILTIN, POSITION]);
        }
        c.extend([opcode(OP_MEMBER_DECORATE, 5), 20, 1, OFFSET, VIEW_PROJECTION_OFFSET]);
        c.extend([opcode(OP_TYPE_VOID, 2), 1]);
        c.extend([opcode(OP_TYPE_FUNCTION, 3), 2, 1]);
        c.extend([opcode(OP_TYPE_FLOAT, 3), 3, 32]);
        c.extend([opcode(OP_TYPE_VECTOR, 4), 4, 3, 4]);
        c.extend([opcode(OP_TYPE_MATRIX, 4), 21, 4, 4]);
        c.extend([opcode(OP_TYPE_INT, 4), 13, 32, 0]);
        c.extend([opcode(OP_CONSTANT, 4), 3, 6, 1.0f32.to_bits()]);
        c.extend([opcode(OP_CONSTANT, 4), 13, 14, 0]);
        c.extend([opcode(OP_CONSTANT, 4), 13, 15, 1]);
        if push {
            c.extend([opcode(OP_CONSTANT, 4), 3, 31, PUSH_START.to_bits()]);
        }
        c.extend([opcode(OP_CONSTANT_COMPOSITE, 7), 4, 7, 6, 6, 6, 6]);
        c.extend([opcode(OP_TYPE_STRUCT, 4), 20, 4, 21]);
        c.extend([opcode(OP_TYPE_RUNTIME_ARRAY, 3), 22, 20]);
        c.extend([opcode(OP_TYPE_POINTER, 4), 23, STORAGE_UNIFORM, 22]);
        c.extend([opcode(OP_TYPE_POINTER, 4), 24, STORAGE_UNIFORM, 20]);
        c.extend([opcode(OP_VARIABLE, 4), 23, 25, STORAGE_UNIFORM]);
        if block {
            c.extend([opcode(OP_TYPE_STRUCT, 3), 9, 4]);
            c.extend([opcode(OP_TYPE_POINTER, 4), 5, STORAGE_OUTPUT, 9]);
        } else {
            c.extend([opcode(OP_TYPE_POINTER, 4), 5, STORAGE_OUTPUT, 4]);
        }
        c.extend([opcode(OP_VARIABLE, 4), 5, 11, STORAGE_OUTPUT]);
        c.extend([opcode(OP_FUNCTION, 5), 1, 10, 0, 2]);
        c.extend([opcode(OP_LABEL, 2), 12]);
        if frame {
            c.extend([opcode(OP_ACCESS_CHAIN, 5), 24, 26, 25, 14]);
        }
        c.extend([opcode(OP_STORE, 3), 11, 7]);
        c.extend([opcode(OP_RETURN, 1)]);
        c.extend([opcode(OP_FUNCTION_END, 1)]);
        c
    }

    fn ops(code: &[u32]) -> Vec<u16> {
        instructions(code).unwrap().iter().map(|i| i.op).collect()
    }

    #[test]
    fn position_z_becomes_w_minus_z_before_the_return() {
        let code = module(false, false, false);
        assert!(is_vertex(&code));
        let twin = mirrored(&code, Some(350_000.0)).unwrap();
        assert!(twin[3] > code[3]);
        let ops = ops(&twin);
        let ret = ops.iter().position(|&op| op == OP_RETURN).unwrap();
        assert_eq!(
            &ops[ret - 6..ret],
            &[
                OP_LOAD,
                OP_COMPOSITE_EXTRACT,
                OP_COMPOSITE_EXTRACT,
                OP_FSUB,
                OP_COMPOSITE_INSERT,
                OP_STORE
            ]
        );
        assert_eq!(instructions(&twin).unwrap().len(), ops.len());
    }

    #[test]
    fn a_block_member_is_reached_through_an_access_chain() {
        let twin = mirrored(&module(true, false, false), None).unwrap();
        let ops = ops(&twin);
        let ret = ops.iter().position(|&op| op == OP_RETURN).unwrap();
        assert_eq!(ops[ret - 7], OP_ACCESS_CHAIN);
        let chain = instructions(&twin)
            .unwrap()
            .into_iter()
            .find(|i| i.op == OP_ACCESS_CHAIN)
            .unwrap();
        assert_eq!(twin[chain.at + 3], 11);
    }

    #[test]
    fn the_precise_ending_reads_the_frame_block_through_the_heap() {
        let code = module(false, true, false);
        let twin = mirrored(&code, Some(350_000.0)).unwrap();
        let ins = instructions(&twin).unwrap();
        let chain = ins
            .iter()
            .filter(|i| i.op == OP_ACCESS_CHAIN)
            .nth(1)
            .unwrap();
        assert_eq!(&twin[chain.at + 3..chain.at + chain.len - 2], &[25, 14, 15][..]);
        let precise = ops(&twin);
        assert!(precise.contains(&OP_FNEGATE));
        assert!(!precise.contains(&OP_EXT_INST));
        assert!(twin.windows(4).any(|w| w[0] == opcode(OP_CONSTANT, 4)
            && w[3] == (1.0f32 / 350_000.0).to_bits()));
        let twin = mirrored(&module(false, true, true), Some(350_000.0)).unwrap();
        assert!(ops(&twin).contains(&OP_EXT_INST));
        let plain = mirrored(&module(false, true, true), None).unwrap();
        assert!(!ops(&plain).contains(&OP_FNEGATE));
    }

    #[test]
    fn a_pixel_shader_is_refused() {
        let mut code = module(false, false, false);
        let entry = instructions(&code)
            .unwrap()
            .into_iter()
            .find(|i| i.op == OP_ENTRY_POINT)
            .unwrap();
        code[entry.at + 1] = 4;
        assert!(!is_vertex(&code));
        assert!(mirrored(&code, None).is_none());
    }
}

