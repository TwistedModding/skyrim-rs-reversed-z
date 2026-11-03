//! The mirrored twin of a SPIR-V vertex shader: before every return of the vertex entry point,
//! the position's z is replaced by w - z. The position is the output decorated `BuiltIn Position`,
//! either a variable or a member of a block.

const OP_ENTRY_POINT: u16 = 15;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
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
const OP_FSUB: u16 = 131;
const OP_RETURN: u16 = 253;
const MODEL_VERTEX: u32 = 0;
const STORAGE_OUTPUT: u32 = 3;
const BUILTIN: u32 = 11;
const POSITION: u32 = 0;
const MAGIC: u32 = 0x0723_0203;

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

/// The twin, or `None` when the shader has no vertex entry point writing a position.
#[allow(clippy::too_many_lines)]
pub(crate) fn mirrored(code: &[u32]) -> Option<Vec<u32>> {
    if code.len() < 6 || code[0] != MAGIC {
        return None;
    }
    let ins = instructions(code)?;
    let mut bound = code[3];
    let mut entry = None;
    let mut position_var = None;
    let mut position_member = None;
    let mut float = None;
    let mut vector = None;
    let mut int = None;
    let mut pointer_out = None;
    let mut pointees = std::collections::HashMap::new();
    let mut variables = Vec::new();
    let mut zero = None;
    let mut first_function = None;
    for i in &ins {
        match i.op {
            OP_ENTRY_POINT if word(code, i, 1) == Some(MODEL_VERTEX) && entry.is_none() => {
                entry = word(code, i, 2);
            }
            OP_DECORATE
                if word(code, i, 2) == Some(BUILTIN) && word(code, i, 3) == Some(POSITION) =>
            {
                position_var = word(code, i, 1);
            }
            OP_MEMBER_DECORATE
                if word(code, i, 3) == Some(BUILTIN) && word(code, i, 4) == Some(POSITION) =>
            {
                position_member = Some((word(code, i, 1)?, word(code, i, 2)?));
            }
            OP_TYPE_FLOAT if word(code, i, 2) == Some(32) => float = word(code, i, 1),
            OP_TYPE_INT if word(code, i, 2) == Some(32) && int.is_none() => {
                int = word(code, i, 1);
            }
            OP_TYPE_VECTOR if word(code, i, 3) == Some(4) && word(code, i, 2) == float => {
                vector = word(code, i, 1);
            }
            OP_TYPE_POINTER => {
                pointees.insert(word(code, i, 1)?, (word(code, i, 2)?, word(code, i, 3)?));
            }
            OP_CONSTANT if Some(word(code, i, 1)?) == int && word(code, i, 3) == Some(0) => {
                zero = word(code, i, 2);
            }
            OP_VARIABLE if word(code, i, 3) == Some(STORAGE_OUTPUT) => {
                variables.push((word(code, i, 2)?, word(code, i, 1)?));
            }
            OP_FUNCTION if first_function.is_none() => first_function = Some(i.at),
            _ => {}
        }
    }
    let entry = entry?;
    let (float, vector) = (float?, vector?);
    let first_function = first_function?;
    let (variable, member) = match (position_var, position_member) {
        (Some(v), _) if variables.iter().any(|(id, _)| *id == v) => (v, None),
        (_, Some((block, member))) => {
            let (id, _) = *variables.iter().find(|(_, pointer)| {
                pointees.get(pointer).is_some_and(|(_, pointee)| *pointee == block)
            })?;
            (id, Some(member))
        }
        _ => return None,
    };
    for (id, (storage, pointee)) in &pointees {
        if *storage == STORAGE_OUTPUT && *pointee == vector {
            pointer_out = Some(*id);
        }
    }
    let mut allocate = || {
        bound += 1;
        bound - 1
    };
    let mut globals = Vec::new();
    let pointer_out = pointer_out.unwrap_or_else(|| {
        let id = allocate();
        globals.extend([opcode(OP_TYPE_POINTER, 4), id, STORAGE_OUTPUT, vector]);
        id
    });
    let index = member.map(|member| {
        let int = int.unwrap_or_else(|| {
            let id = allocate();
            globals.extend([opcode(OP_TYPE_INT, 4), id, 32, 0]);
            id
        });
        if member == 0 && let Some(zero) = zero {
            zero
        } else {
            let id = allocate();
            globals.extend([opcode(OP_CONSTANT, 4), int, id, member]);
            id
        }
    });
    let mut out = code[..first_function].to_vec();
    out.extend(&globals);
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
                        let id = allocate();
                        out.extend([
                            opcode(OP_ACCESS_CHAIN, 5),
                            pointer_out,
                            id,
                            variable,
                            index,
                        ]);
                        id
                    }
                    None => variable,
                };
                let (loaded, z, w, flipped, inserted) =
                    (allocate(), allocate(), allocate(), allocate(), allocate());
                out.extend([opcode(OP_LOAD, 4), vector, loaded, pointer]);
                out.extend([opcode(OP_COMPOSITE_EXTRACT, 5), float, z, loaded, 2]);
                out.extend([opcode(OP_COMPOSITE_EXTRACT, 5), float, w, loaded, 3]);
                out.extend([opcode(OP_FSUB, 5), float, flipped, w, z]);
                out.extend([
                    opcode(OP_COMPOSITE_INSERT, 6),
                    vector,
                    inserted,
                    flipped,
                    loaded,
                    2,
                ]);
                out.extend([opcode(OP_STORE, 3), pointer, inserted]);
            }
            _ => {}
        }
        out.extend(&code[i.at..i.at + i.len]);
    }
    out[3] = bound;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OP_TYPE_VOID: u32 = 19;
    const OP_TYPE_FUNCTION: u32 = 33;
    const OP_CONSTANT_COMPOSITE: u32 = 44;
    const OP_LABEL: u32 = 248;
    const OP_TYPE_STRUCT: u32 = 30;

    fn module(block: bool) -> Vec<u32> {
        let mut c = vec![MAGIC, 0x0001_0000, 0, 20, 0];
        c.extend([opcode(17, 2), 1]);
        c.extend([opcode(14, 3), 0, 1]);
        c.extend([opcode(OP_ENTRY_POINT, 5), MODEL_VERTEX, 10, 0x006e_6961, 11]);
        if block {
            c.extend([opcode(OP_MEMBER_DECORATE, 5), 9, 0, BUILTIN, POSITION]);
        } else {
            c.extend([opcode(OP_DECORATE, 4), 11, BUILTIN, POSITION]);
        }
        c.extend([OP_TYPE_VOID | (2 << 16), 1]);
        c.extend([OP_TYPE_FUNCTION | (3 << 16), 2, 1]);
        c.extend([opcode(OP_TYPE_FLOAT, 3), 3, 32]);
        c.extend([opcode(OP_TYPE_VECTOR, 4), 4, 3, 4]);
        c.extend([opcode(OP_CONSTANT, 4), 3, 6, 1.0f32.to_bits()]);
        c.extend([OP_CONSTANT_COMPOSITE | (7 << 16), 4, 7, 6, 6, 6, 6]);
        if block {
            c.extend([OP_TYPE_STRUCT | (3 << 16), 9, 4]);
            c.extend([opcode(OP_TYPE_POINTER, 4), 5, STORAGE_OUTPUT, 9]);
        } else {
            c.extend([opcode(OP_TYPE_POINTER, 4), 5, STORAGE_OUTPUT, 4]);
        }
        c.extend([opcode(OP_VARIABLE, 4), 5, 11, STORAGE_OUTPUT]);
        c.extend([opcode(OP_FUNCTION, 5), 1, 10, 0, 2]);
        c.extend([OP_LABEL | (2 << 16), 12]);
        if block {
            c.extend([opcode(OP_TYPE_INT, 4), 13, 32, 0]);
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
        let code = module(false);
        assert!(is_vertex(&code));
        let twin = mirrored(&code).unwrap();
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
        let twin = mirrored(&module(true)).unwrap();
        let ops = ops(&twin);
        let ret = ops.iter().position(|&op| op == OP_RETURN).unwrap();
        assert_eq!(ops[ret - 7], OP_ACCESS_CHAIN);
        assert!(ops.contains(&OP_TYPE_POINTER));
        let chain = instructions(&twin)
            .unwrap()
            .into_iter()
            .find(|i| i.op == OP_ACCESS_CHAIN)
            .unwrap();
        assert_eq!(twin[chain.at + 3], 11);
    }

    #[test]
    fn a_pixel_shader_is_refused() {
        let mut code = module(false);
        let entry = instructions(&code)
            .unwrap()
            .into_iter()
            .find(|i| i.op == OP_ENTRY_POINT)
            .unwrap();
        code[entry.at + 1] = 4;
        assert!(!is_vertex(&code));
        assert!(mirrored(&code).is_none());
    }
}
