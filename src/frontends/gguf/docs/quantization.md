# GGUF weight formats, unpacking and execution precision

The GGUF frontend separates interleaved GGML blocks into OpenVINO weight, scale and optional
zero-point Constants. This guide covers their decoding, stored element types, storage costs
and conversion tradeoffs. File quantization, frontend storage, inference arithmetic and
KV-cache precision are separate choices; device compilation determines the execution
representation.

The implementation is in [`gguf.cpp`](../src/quant/gguf.cpp),
[`gguf_quants.cpp`](../src/quant/gguf_quants.cpp) and [`weights.cpp`](../src/quant/weights.cpp).

## Storage and graph conventions

The tables below describe the **native `.gguf` input path before device compilation**. The
[supplied-decoder path](#supplied-decoder-and-raw-byte-differences) has additional conversions.
Mixed checkpoint labels such as `Q4_K_M` select different formats for different tensors; they
do not define another block layout.

For a logical weight shape `[..., K]` and group size `G`, the decompression graph views codes
as `[..., K/G, G]` and scales/zero points as `[..., K/G, 1]`. It computes
`weight = (code - zero_point) * scale`, or `code * scale` for symmetric formats, then reshapes
back to `[..., K]`. Codes, scales and zero points are separate Constants. GGUF type identifiers
select the import algorithm; they do not become new OpenVINO element types.

Sub-byte Constants remain packed: `u1` holds eight binary codes per byte, with the first element
in the most significant bit. `u2` holds four codes per byte, and `i4`/`u4` hold two per byte,
with the first element in the least significant bits. The loader's `u32` tensors for
Q4_1/Q4_K are backing buffers for eight nibbles per word; their graph Constants use `u4`.
Unpacking into `u2` or `i4` therefore does not mean allocating one byte per weight.

For ternary embedding row lookups, the frontend views the same `u2` buffer as `u8` bytes,
with four codes in each byte. It gathers the requested code bytes and scales before
unpacking the selected rows with integer shifts and masks. The dictionary keeps its
original two-bit storage; it does not expand to an entire floating-point table. The
selected codes use I32 arithmetic for shifts, then the original F16 dequantization.
Matrix multiplication consumers keep their `u2` representation, including tied weights.

The usual graph is `Convert -> [Subtract] -> Multiply -> Reshape -> Convert(F32)`. Decompression
uses the scale's element type, normally F16. Q6_K, Q8_1 and formats with fractional F16 zero points
and code widths of at least four bits instead perform subtraction/multiplication in F32.
Q8_K uses F32 scales and thus F32 arithmetic. MXFP4 uses an F16 multiply. A final F32 output
does not undo rounding in earlier F16 operations, and does not imply that a full F32 weight
buffer is stored by the frontend.

Affine formats replace the source's additive offset with a fractional F16 zero point.
The scale products and zero-point ratios can round or overflow; a zero scale cannot retain
a nonzero offset. Fractional zero points can also prevent compressed FullyConnected fusion.
Integer-zero-point requantization avoids that fusion limitation where supported, but changes
the weight values and adds quantization error.

## Native weight formats

`W`, `S` and `Z` below mean code, scale and zero-point payloads. Byte counts include all three
but exclude object metadata, shape Constants, allocator alignment and file padding. Each row
compares the same number of logical weights; a negative change means smaller storage.

The type columns describe extracted tensors; their decompression or Convert outputs are F32.

| Format | Weights per GGUF block | GGUF bytes | Weight type (`W`) | Scale type (`S`) | Zero-point type (`Z`) | Native OpenVINO bytes (`W + S + Z`) | Change |
|---|---:|---:|---|---|---|---:|---:|
| F32 | 1 | 4 | `f32` | — | — | 4 | 0% |
| F16, BF16 | 1 | 2 | `f16`, `bf16` respectively | — | — | 2 | 0% |
| Q1_0 | 128 | 18 | `u1` | `f16` | — | `16 + 2` = 18 | 0% |
| Q2_0 (per-64) | 64 | 18 | `u2` | `f16` | `u8` | `16 + 2 + 1` = 19 | +5.6% |
| PQ2_0, ternary codes | 128 | 34 | `u2` | `f16` | scalar `f16` | 34, plus 2 per tensor | 0% group payload |
| PTQ1_0 | 128 | 28 | `u2` | `f16` | scalar `f16` | 34, plus 2 per tensor | +21.4% group payload |
| Q2_K | 256 | 84 | `u2` | `f16` | `f16` | `64 + 32 + 32` = 128 | +52.4% |
| Q3_K | 256 | 110 | `i4` | `f16` | — | `128 + 32` = 160 | +45.5% |
| Q4_0 | 32 | 18 | `i4` | `f16` | — | `16 + 2` = 18 | 0% |
| Q4_1, integer zero point | 32 | 20 | `u4` | `f16` | `u8` | `16 + 2 + 1` = 19 | -5.0% |
| Q4_1, fractional zero point | 32 | 20 | `u4` | `f16` | `f16` | `16 + 2 + 2` = 20 | 0% |
| Q4_K, integer zero point | 256 | 144 | `u4` | `f16` | `u8` | `128 + 16 + 8` = 152 | +5.6% |
| Q4_K, fractional zero point | 256 | 144 | `u4` | `f16` | `f16` | `128 + 16 + 16` = 160 | +11.1% |
| Q5_0 | 32 | 22 | `i8` | `f16` | — | `32 + 2` = 34 | +54.5% |
| Q5_1 | 32 | 24 | `i8` | `f16` | `f16` | `32 + 2 + 2` = 36 | +50.0% |
| Q5_K, ordinary matrix | 256 | 176 | `i8` | `f16` | — | `256 + 16` = 272 | +54.5% |
| Q5_K, fractional zero point | 256 | 176 | `u8` | `f16` | `f16` | `256 + 16 + 16` = 288 | +63.6% |
| Q6_K | 256 | 210 | `i8` | `f16` | — | `256 + 32` = 288 | +37.1% |
| Q8_0 | 32 | 34 | `i8` | `f16` | — | `32 + 2` = 34 | 0% |
| Q8_1 | 32 | 36 | `i8` | `f16` | — | `32 + 2` = 34 | -5.6% |
| Q8_K | 256 | 292 | `i8` | `f32` | — | `256 + 4` = 260 | -11.0% |
| MXFP4 | 32 | 17 | `f4e2m1` | `f8e8m0` | — | `16 + 1` = 17 | 0% |

Ordinary native Q4_1/Q4_K matrix weights use integer zero points by default.
Ordinary native Q5_K matrix weights use symmetric i8 with one F16 scale per 32 weights
and no zero point. Source values are decoded and requantized using the group maximum
absolute value; this adds rounding error and changes the grid from the former asymmetric u8
representation. Conversion uses a 256-value scratch buffer per source block. The converted
weight/scale payload is 2.86% smaller than the former u8/integer-zero-point payload.
Group size remains 32 for all devices.
`token_embd.weight` and `output.weight` use fractional zero points for these affine formats;
Q5_1 and Q2_K also use fractional zero points. See [accuracy controls](#accuracy-controls)
to select fractional zero points for all Q4_1/Q4_K tensors.
Q2_0 uses a `u8` zero point of 1. PQ2_0/PTQ1_0 use F16 zero points of 1,
including embeddings.
The frontend prints an accuracy notice once per process for each requantization approximation.

PTQ1_0 adds six code bytes per 128 weights. For PQ2_0/PTQ1_0 containing only codes
0, 1 and 2, their native storage is identical, at `34/128` bytes per weight
(2.125 bits per weight including scales), plus one two-byte zero-point Constant per tensor.
Code 3 retains a grouped Constant and the existing GPU fallback, using 36 bytes per group.
Projections sharing an activation input retain grouped zero points if any projection needs
that fallback, including a singleton group. This keeps their layouts compatible with
horizontal fusion without changing plugins; all-ternary projection groups remain scalar.

## Floating-point formats

F32, F16 and BF16 tensors are views of the memory-mapped file, without block unpacking or
quantization metadata. F16/BF16 Constants receive a Convert to F32 for operation translators.
There is no additional code quantization. Compilation may materialize a converted buffer,
so the original two-byte storage is not an upper bound on compiled weight memory. Auxiliary
integer tensors and F64 tensors likewise have no quantized block decomposition.

## Binary and ternary formats

### Q1_0

A block is an F16 scale followed by 16 bytes of binary codes for 128 weights. Import reverses
the bits in each byte from GGML's least-significant-bit-first order to OpenVINO's
most-significant-bit-first `u1` order. The scale is copied unchanged. Eight weights remain
packed per byte, without a zero-point tensor or requantization.

The graph converts bits to F16, computes `2 * bit - 1`, multiplies by the per-128 F16 scale,
reshapes to the logical weight dimensions and converts to F32. Forming the sign before
applying the scale avoids doubling the stored scale and overflowing large finite F16 scales.
The sign expression adds only two scalar Constants shared across groups. Both native and
raw-byte input paths preserve `u1`, including embedding/output tensors.

Device plugins need compressed binary matrix and lookup kernels to preserve this storage
through compilation. The current CPU compressed FullyConnected and Gather paths do not
support `u1`, so compilation expands these weights to floating point. This can substantially
increase execution and compilation memory despite the compact frontend representation.

### Q2_0 and PQ2_0

Q2_0 stores `F16 scale | 16 code bytes` for 64 weights. PQ2_0 (type 142) stores
`F16 scale | 32 code bytes` for 128 weights. Both already pack four two-bit codes per byte in
OpenVINO element order, so the code bytes are copied directly. The scale is separated.
Q2_0 uses a `u8` zero point of 1 per group; PQ2_0 uses an FP16 scalar zero point of 1
for strictly ternary tensors. Tensors containing
code 3 retain grouped FP16 zero points. Reconstruction is `(code - 1) * scale`;
codes 0, 1 and 2 represent -1, 0 and +1. A code of 3 reconstructs +2.

No code or scale requantization is introduced.
Legacy files can also use type 42 (`Q2_0`) for per-128 blocks. Native loading identifies
that layout using dimensions and aligned tensor spans and maps it to PQ2_0. Raw-byte callers
must provide the actual format because they do not supply those descriptor spans.

### PTQ1_0

PTQ1_0 (type 143) stores `26 dense-trit bytes | F16 scale` for 128 weights. The first 16 bytes
encode five trits each, the next eight encode five each, and the last two encode four each:
`16*5 + 8*5 + 2*4 = 128`. This scaled-byte encoding is decoded in the following order:

```text
for (start, count, trits) in [(0, 16, 5), (16, 8, 5), (24, 2, 4)]:
    for n in 0 .. trits-1:
        for m in 0 .. count-1:
            q = (payload[start+m] * 3**n) modulo 256
            code = (q * 3) >> 8
            append code to packed u2 output
```

The modulo operation reflects the implementation's `uint8_t` intermediate; replacing this
with ordinary base-3 digit extraction would change the decoding. Import writes 32 bytes of
`u2` codes, copies the F16 scale and adds the same zero point of 1 as PQ2_0. There is no
decoding table or dense-trit decoder left in the IR; its graph is the usual grouped
Convert/Subtract/Multiply/Reshape decomposition.

The conversion preserves trit values but expands their dense storage to two bits per code.
It also requires more import work than PQ2_0's direct copy.

## Integer formats

### Q4_0

A block holds an F16 scale and 16 bytes of four-bit codes. GGML's low nibbles describe the
first 16 weights and its high nibbles the next 16. Import reorders them into adjacent-element
packing, then XORs each byte with `0x88`. This converts unsigned codes 0..15 into signed
`i4` codes -8..7. The graph multiplies by the copied scale with group 32, without a zero point.

Storage size and code values are preserved after centering. Import still performs a reorder;
the F16 multiply and device arithmetic can round differently from a GGML kernel.

### Q5_0 and Q5_1

Both formats combine a four-byte high-bit mask with 16 bytes of low nibbles to reconstruct
32 five-bit codes: `q = low_nibble | (high_bit << 4)`.

Q5_0 stores one F16 scale and centers `q` by subtracting 16 at import, producing `i8` codes
-16..15 with group 32. Q5_1 stores an F16 scale `d` and an F16 additive offset `m`, keeps
`i8` codes 0..31, and stores the F16 zero point `-m/d`. Its graph computes `(q - zero_point) * d`
in F32; import uses zero point 0 when `d` is zero.

Both widen five-bit codes to one byte per weight without code requantization. Q5_1 has the
affine scale/zero-point limitations described in [graph conventions](#storage-and-graph-conventions).

### Q3_K

A 256-weight block contains 32 high-mask bytes, 64 low-code bytes, 12 packed sub-scale bytes
and an F16 scale `d`. Import reconstructs the signed three-bit code as
`low_two_bits - (high_mask_bit ? 0 : 4)`, reorders it into logical element order, and stores
it as `i4` (-4..3). Six-bit sub-scales are unpacked and centered by subtracting 32. Each
16-weight group receives an F16 scale `d * signed_sub_scale`.

There is no code requantization, but three-bit codes widen to four bits and compact scale
metadata becomes 16 F16 scales. Products stored as F16 can round or overflow.

### Q6_K

A block contains 128 low-nibble bytes, 64 high-two-bit bytes, 16 signed byte sub-scales and
an F16 scale `d`. Import joins the six-bit codes, subtracts 32, and writes `i8` (-32..31) in
logical order. Sixteen F16 scales hold `d * signed_sub_scale` for groups of 16. The graph
explicitly converts codes and scales to F32 before multiplying.

Code values are preserved, but widening to eight bits and expanding sub-scales increases storage.
F32 graph arithmetic avoids an F16 multiply, but the combined scales have already been rounded
to F16 at import.

### Q8_0, Q8_1 and Q8_K

Q8_0 directly copies 32 `i8` codes and one F16 scale per block. There is no code expansion,
requantization or extra metadata; its usual decompression multiply is F16.

Q8_1 (type 9) uses the current ggml layout `F16 d | F16 s | 32 i8 codes`, totaling 36 bytes.
Import copies `d` and the codes unchanged and omits `s`, a precomputed `d * sum(codes)` used
by ggml dot-product kernels. It is not an additive weight offset or zero point. OpenVINO
stores 34 bytes per block, 5.6% less than the GGUF payload. Legacy 40-byte blocks with F32
`d` and `s` are not supported; the frontend does not infer that layout from padded tensor spans.

Both input paths keep the original per-32 F16 scales for every Q8_1 weight name, including
embedding/output tensors. The graph converts codes and scales to F32 before multiplying,
preserving finite reconstructed weight values exactly without F16 multiplication rounding or
requantization. The stored Constants remain i8 and F16; no full F32 weight Constant is required
at import. Device compilation can still expand the weights or select different execution
precision, so the compact payload and exact graph reconstruction do not guarantee runtime
memory usage or bit-exact inference.

Q8_K copies 256 `i8` codes and an F32 scale unchanged. Each GGUF block contains 256 code bytes,
four scale bytes and 32 bytes of precomputed sums (`16` values of type `i16`), totaling 292
bytes. OpenVINO omits the sums, leaving 260 bytes: a reduction of `32/292`, or 11.0%.

Each omitted sum is `bsums[j] = sum(codes[16*j : 16*j+16])`. These sums accelerate GGML's
quantized dot products but are not needed by the decompression graph. No weight information
or precision is lost by omitting them: reconstruction remains `weight[i] = scale * code[i]`
in F32, without requantization. The sums can be regenerated exactly from the retained codes
if a future kernel or GGUF exporter needs them. Preserving them would add unused storage to
the current representation. OpenVINO does not reproduce GGML's partial-sum kernel through
this representation.

## Affine K formats

### Q2_K

A block is `16 scale/min bytes | 64 code bytes | F16 d | F16 dmin`. Each scale/min byte
contains a four-bit sub-scale `s` and a four-bit sub-minimum `m`. Import reorders the two-bit
codes into packed `u2` element order. For each 16-weight group it stores F16 scale `d*s`
and F16 zero point `(dmin*m)/(d*s)`, or zero when the denominator is zero.

Codes stay two-bit without requantization, but the expanded metadata adds storage and has the
common affine limitations. Decompression arithmetic is F16; `u2` also limits device fusion.

### Q4_K and Q5_K

Both use 256-weight blocks with F16 `d`/`dmin` and 12 bytes packing eight six-bit sub-scales
and minima. Q4_K has 128 code bytes; Q5_K adds 32 high-bit bytes. Import extracts each
32-weight group's scale `sc = d*s` and subtractive offset `offset = dmin*m`, giving source
values `sc*q - offset`.

With a **fractional zero point**, Q4_K reorders and preserves the four-bit codes in `u4`;
Q5_K reconstructs five-bit codes 0..31 and stores them in `u8`. Both store F16 `sc` and
F16 `offset/sc` (zero for a zero scale), and their graph arithmetic is F32. Both have the
common affine limitations; Q5_K additionally widens its code payload.

With an **integer zero point**, Q4_K first reconstructs 32 source values in F32, then
requantizes onto a new grid whose range includes zero. It uses 16 levels in packed `u4`:
min/max initialization, an integer zero point, and a least-squares scale refinement. Candidate
scales are evaluated after F16 rounding. The importer retains the original codes with a
rounded integer zero point unless a candidate reduces squared error without increasing
maximum absolute error.

Ordinary Q5_K matrix weights use a symmetric 255-level i8 grid. Each 32-value group uses `F16(max(abs(value))/127)` and codes rounded against that stored scale,
clamped to [-127,127]. There is no zero-point tensor. A skewed range can have a coarser
grid than asymmetric u8, so this representation changes reconstruction error even though
both paths requantize once. Embedding/output and other non-matrix Q5_K tensors retain
the existing u8 handling.

Including zero in a one-sided group coarsens the available grid. Q5_K collapses a group to
zero if its new F16 scale rounds to zero. The symmetric path rejects nonfinite source
values and scales that exceed the F16 range.

### Q4_1

Q4_1 stores `F16 d | F16 m | 16 code bytes` for 32 weights, with values `d*q + m`.
The fractional path reorders its `u4` codes, copies `d` and stores F16 zero point `-m/d`,
then decompresses in F32 with the common affine limitations. The integer path uses the Q4_K
group requantizer described above with offset `-m`, reducing metadata by one byte per group.

## MXFP4

A block holds one E8M0 scale byte and 16 bytes of four-bit codes for 32 weights. Import
reorders the split low/high nibbles into element order and separates an `f4e2m1` weight
Constant from an `f8e8m0` scale Constant. OpenVINO's `f4e2m1` values are half the GGML lookup
values, while its E8M0 scale is twice the GGML scale; their product represents the same value.

The payload remains 17 bytes with no code requantization or host float expansion. The graph
converts both operands to F16, multiplies and converts the reshaped result to F32. F16 range
and arithmetic, and the device's support for these compressed element types, still matter.

## Supplied-decoder and raw-byte differences

The native loader supplies pre-extracted tensors to `make_weight_node`. A supplied decoder
can instead pass raw GGML bytes, a type name and a logical shape. The raw-byte overload
currently differs as follows:

| Case | Raw-byte representation | Drawback |
|---|---|---|
| Q4_1, Q5_1 | Full F32 Constant decoded with `d*q + m` before other format handling | 128 bytes per 32 weights: +540% for Q4_1, +433.3% for Q5_1 |
| Q1_0, including embedding/output | Packed `u1` plus per-128 F16 scales, as in native loading | Requires plugin binary decompression support to remain packed at execution |
| Q8_1, including embedding/output | Original `i8` codes plus per-32 F16 scales, with F32 graph multiplication, as in native loading | Device compilation determines execution storage and precision |
| Quantized `token_embd.weight` / `output.weight`, except Q1_0/Q8_1 and the Q4_1/Q5_1 cases above | Channel-wise Q8_0_C: `i8` codes with one F16 scale per row | Additional lossy quantization; low-bit sources usually expand |
| Q5_K ordinary matrices | The same channel-wise Q8_0_C representation | Additional lossy quantization; different scale grouping from native Q5_K |
| Q4_K/Q5_K/Q6_K selected for Q8_0_C | Stream original F32-dequantized rows directly into channel-wise quantization | Per-worker F32 row scratch; avoids allocating a full temporary F32 matrix |
| Rank greater than two MXFP4 experts | Packed `u8` blocks retaining scale/code interleaving; expert selection and decompression happen on the graph | Different graph layout; device fusion and temporary storage depend on expert execution |

For Q8_0_C, each row uses `d = max(abs(values))/127`, stored as F16, and signed codes rounded
from `values/d`. A zero row has zero scale. A matrix of `R` rows and `K` columns stores
`R*K + 2*R` bytes. Other selected quantized sources are first extracted and then dequantized
one row at a time for this conversion. Plain F32/F16/BF16 return before channel-wise
requantization. Raw Q8_K has no implemented fill case, unlike the native Q8_K path.

## Accuracy controls

For strict native CPU comparisons, use F32 inference, disable dynamic activation quantization,
and select the reference's KV-cache precision (the decoder fixture suite uses F16):

```python
compiled = core.compile_model(model, "CPU", {
    "INFERENCE_PRECISION_HINT": "f32",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
    "KV_CACHE_PRECISION": "f16",
})
```

Set `OV_GGUF_Q4_K_ZP_F16=1` **before starting the process** to preserve fractional Q4_K/Q4_1
zero points instead of their default u4 requantization. It is cached on first use; changing the
environment later in a notebook does not switch an already initialized process. The option
does not make every quantized format lossless or guarantee device support for fractional zero points.

llama.cpp's quantized kernels can quantize activations even when OpenVINO's dynamic activation
quantization is disabled. Report comparisons against quantized CPU arithmetic separately from
F32 arithmetic on the same represented weights. Improved greedy-token agreement can coexist
with worse full-logit error; use both metrics and the [acceptance criteria](testing.md#acceptance).

## Memory and packaging limits

For `N` weights of a uniform block format, multiply its OpenVINO bytes per block by
`N / weights_per_block`. Sum over tensor formats for a mixed checkpoint. This estimates
frontend payload storage, including per-group metadata.

Native quantized loading allocates independent buffers while reading the memory-mapped
source. Source pages and output buffers can coexist during import; the loader requests
eviction of processed source ranges, but this is not a guarantee about peak resident memory.
Splitting or reordering projections can also require copies. Constants built from the
extracted buffers share those buffers rather than copying each one again.

During compilation a device may fuse decompression into its matrix kernel, reorder packed
weights, or expand them to floating point. Full F16 and F32 expansions require `2*N` and
`4*N` bytes respectively, potentially alongside packed storage. In particular, CPU compressed
expert matmul does not support `u2` in this implementation, so Q2 MoE weights can expand.
Weight reorders and temporary buffers add to these costs; reducing context length does not
remove them. KV caches, activation buffers and Hadamard transforms are separate from weight
payload estimates. See [checkpoint-specific limits](supported_models.md#runtime-limitations).
