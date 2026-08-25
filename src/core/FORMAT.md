# VK Core binary formats: FB v5 / PB v5

Byte-exact specification of the standalone core serialization (`vk_graph_format`).
All integers are little-endian. Every read is bounds-checked: truncated or
corrupt blobs fail loudly with `[GPU] vk_graph_format: ...`.

## Container

| Blob | Magic | Payload |
|---|---|---|
| FB (Float Binary) | `VKFB0001` | one ir_graph |
| PB (Parallel Binary) | `VKPB0001` | u32 version, u32 graph count, then per graph: u32 fb_size + FB bytes |

## FB body (after magic)

```
u32  format_version                     // 5
u32  num_tensor_shapes                  ┐
u32  num_constants                      │
u32  num_quant_constants                │ counts
u32  num_nodes                          │
u32  num_inputs, num_outputs            ┘
per tensor shape : str id, u32 rank, rank × u64 dims
per constant     : str id, u32 count, count × f32
per quant const  : str id, u32 gguf_type, u32 byte_len, byte_len × u8
per node         :
    u8   op code                        // see table
    str  id
    u8   matmul_transpose_b
    f32  alpha                          // leaky_relu slope / rms eps / pad fill
    u32  axis                           // concat / softmax / reduce / cache pos
    u64v transpose_order                 // permutation
    u64v pool.kernel, pool.strides,
         pool.pads_begin, pool.pads_end  // 4 vectors
    u32 num_inputs, inputs × str
inputs × str, outputs × str            // port order
```

## Op codes (u8)

| code | op | code | op |
|---|---|---|---|
| 0 | parameter | 15 | reduce_max |
| 1 | constant | 16 | gelu |
| 2 | result | 17 | swiglu |
| 3 | relu | 18 | quick_gelu |
| 4 | add | 19 | rms_norm |
| 5 | max_pool | 20 | pad |
| 6 | avg_pool | 21 | crop |
| 7 | convolution | 22 | causal_softmax |
| 8 | matmul | 23 | rope |
| 9 | mul | 24 | cache_write |
| 10 | sub | 25 | argmax |
| 11 | div | | |
| 12 | sigmoid | | |
| 13 | tanh | | |
| 14 | leaky_relu | | |

Codes are append-only: new ops take the next free number, existing blobs stay
readable. Version bumps only when the LAYOUT changes (v2 alpha, v3 quant
payloads, v4 axis+perm, v5 pads_end).

## Semantics pinned by tests

- elementwise: same-shape or constant-broadcast (expanded by the core);
  dynamic mismatches are errors
- softmax/reduce: arbitrary axis; softmax keeps shape, reduce drops it
- rms_norm/softmax lines: [outer, len, inner] indexing
- rope: x `[...,H,D]` even D; cos/sin `x_dims[:-2] + [D/2]`; halves convention
- cache_write: functional — out = cache with rows `[pos, pos+L)` replaced
- conv: exactly 3 inputs (data, weights, bias); dilation = 1
