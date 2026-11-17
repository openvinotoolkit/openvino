# 08. Inline vISA assembly: when, how, and what breaks

The DG2 raw-kernel gate of chapter 06 could not be reached with OpenCL C alone: IGC sank loads to just before their use, unrolled operand packing into word-movs, and semantically equivalent source variants swung by ±40%. The final candidates bundled DPAS, LSC loads, register packing and SLM writes into **vISA inline assembly** to specify the messages and their order directly, leaving only the physical GRF placement to IGC. This chapter records the method and the things that actually broke along the way.

> Scope: this asm belongs to the DG2 standalone experimental kernels (`test/sdpa_ocl_xe_hpg/s7a/perf/opt/short_diag_20261003/raw_stage/`). It was not integrated into the product `sdpa_ocl.cl`. The user's decision was "meet the performance target first, then decide on an OpenCL C/intrinsic conversion in the maintainability review". The "what breaks" list below should therefore also be read as a *pre-integration review checklist*.

## 8.1 Why asm was needed (evidence)

| Evidence | Source |
|---|---|
| OpenCL C `block_read`/gather → IGC sinks the loads to just before DPAS: `loads → sync.allwr → dpas` repeats every step, 0 overlap between steps | ISA reading (chapter 07 §7.4) |
| `-cl-intel-no-prera-scheduling` and `__asm__ volatile("":::"memory")` cannot change the schedule (the former gives identical ISA; the latter unrolls the block read into scalars and spills) | HANDOFF §5 |
| sdpa_micro is fused nGEN native: global-next load → SLM operand → DPAS → next rearrangement → SLM write, no separate K'/V' | micro final ISA analysis |
| The best raw-only candidate built in pure OpenCL was ~10–30% slower than micro; the gap closed after native packing + control of the load order | chapter 06 §6.3 |
| The word-movs of VNNI packing are ~80 instructions per operand (OpenCL); reducible to two native `mov <2>:uw` | ISA comparison |

Decision criterion: use asm **only after measurement has confirmed that "schedule/message/packing instructions" are the bottleneck** (source-variant ablation + ISA confirmation). If the bottleneck is memory bandwidth or the algorithm's structure, asm does not help.

## 8.2 Syntax and idioms (from the actual source)

### Structure: `__asm__ volatile("{ ... }" : outputs : inputs)`

```c
SDPA_OCL_INLINE uint16 raw_v_pair_pack(uint8 x, uint8 y) {
    uint16 p;
    __asm__ volatile("{\n"
        ".decl PW v_type=G type=uw num_elts=256 alias=<%0,0>\n"
        ".decl XW v_type=G type=uw num_elts=128 alias=<%1,0>\n"
        ".decl YW v_type=G type=uw num_elts=128 alias=<%2,0>\n"
        "mov (M1_NM,8) PW(0,0)<2> XW(0,0)<16;8,2>\n"      // interleave the halfwords of two tokens into one dword
        "mov (M1_NM,8) PW(0,1)<2> YW(0,0)<16;8,2>\n"
        /* ... 16 pairs ... */
        "}\n" : "=rw"(p) : "rw"(x), "rw"(y));
    return p;
}
```

- `.decl NAME v_type=G type=<ud|uw|uq|f|hf...> num_elts=N [align=wordx32] [alias=<%k,offset>]`: virtual register declaration. `alias=<%k,off>` *reinterprets* a byte offset of a C variable (`%k`) as another type. This is the key tool for VNNI/transposed rearrangement.
- A local `.decl` is valid only inside its `{ }` block. `align=wordx32` is GRF alignment (needed when a raw operand is used as a payload).
- Constraints: `"=rw"` output, `"rw"` input, `"+rw"` input/output (accumulator). Values go into virtual registers, and IGC allocates the physical GRFs.
- Region syntax `<vstride;width,hstride>`: `XW(0,0)<16;8,2>` spreads 8 halfwords at stride 2 over 8 lanes. `(M1_NM,8)` is execution size 8, mask ignored (NM).
- Predicate: `.decl P v_type=P num_elts=1` + `cmp.ne (M1_NM,1) P ...` + `(P) jmp (M1_NM,1) LABEL`. Labels (`KQWORK:`) are also written directly inside the asm. **A uniform branch inside asm was safer than an OpenCL C branch** (§8.4-4).

### Memory/DPAS instructions

```
lsc_load.ugm.ca.ca (M1,8)  %9:d32x8  flat[%13]:a64           // global, vector of 8 row dwords (per-lane address)
lsc_load.slm       (M1_NM,1) P0:d32x64t flat[%12+0]:a32       // SLM block, transposed(t), 256 B
dpas.hf.hf.8.8     (M1,8) %0.0 %0.0 P0.0 PD0(0,0)             // C = C + A*B, depth 8 repeat 8
dpasw.hf.hf.8.8    ...                                        // "dpas.w." is a syntax error
```

## 8.3 Verified usage patterns

1. **The whole KQ in one asm block**: the depth loop (bundle of 8×8 DPAS), the next-K load and the register transpose sit in one block (`raw_kq_original_whole`). Per-chunk asm → whole-KQ asm: 179.5 → 171.6 µs (seq512, measured on the early raw candidate).
2. **SV step `raw_sv_step`**: `d32x64t` transposed load of P from SLM → 8 DPAS and the next V load (position toggled by `RAW_SV_PIPE`) → next uw packing.
3. **K d64×4 gather** (`kv_k_d64`): reading K with a `d64x4` gather + uq register transpose instead of `d32x8` kept the *same* 32 B/lane and 16K×32Q DPAS while preventing the compiler from chopping the transpose into small pieces. seq512 138.6 → 129.3 µs (micro 136.7). This single change was the turning point that brought the raw-only candidate close to micro. It is evidence that **native message grouping changes the compiler's lowering**.
4. **clamped whole-SV**: instead of handling the SV tail of a partial tile with an OpenCL branch, clamp the address to a valid row (`min(addr, last_valid)`), always run the full DPAS, and make invalid keys contribute 0 through the mask (P=0). Large improvement: h64 seq1033 base 146.3 → 134.4 µs, h96 seq2048 323.7 → 232.0 µs (micro 363.7). The effect came not only from the number of tail iterations but from **the code generation of the branch and the operand marshalling itself** (this was not proven separately, so do not write "because of fewer tail iterations").
5. **PV address algebra**: computing a clamp and a stride product for every 16 tokens was replaced by a shared column/first/last byte address + an unsigned min with fixed offsets (`h128_q16wide_address`). An *address-computation-only* change with asm/DPAS/geometry/barrier/math unchanged: the h128 2/2 seq256 gap went from +4.06% → +1.70%.

## 8.4 What actually broke (recurrence-prevention checklist)

| # | Symptom | Cause | Rule |
|---|---|---|---|
| 1 | Passing the SLM address as `"rw.u"` (forced uniform) gave wrong output (large maxabs). Isolated: `asm_mad` alone PASS, `asm_slm_read` alone FAIL | An SLM address that differs per subgroup was declared uniform | **Subgroup-varying values use `rw`**. When in doubt, split the asm bundle by function and add/remove one piece at a time to isolate the wrong output. A memory clobber did not fix this wrong output |
| 2 | 64-bit address `uq` ADD gave wrong output (maxabs 1.854) | Native ISA lowers the low word of a varying 64-bit ADD in scalar form — carry handling is missing | For 64-bit address addition use `addc` (carry low/high) or precompute it in C. "explicit ud strided low-word ADD" does not handle the carry and cannot be a product candidate |
| 3 | Changing the vector address computation of head-272 broke only the upper 16-channel band (maxabs 2.432) | The exact IGC payload-lowering cause is **unidentified**. A Q-only control FAILs; a fixed `+32` upper load PASSes | Do not generalise a variant whose cause is unknown. `*_exact_v2`, which keeps the original vector pointer and skips only the *absent* upper band with a uniform native branch, PASSes. The old `>256 *_exact` is discarded |
| 4 | An OpenCL C outer SV skip guard gave 45 NaN/Inf at seq 257 (`raw_h80_vactive`); seq256 PASSes | Compiler-controlled reordering of control flow around barriers/SLM | Put guards in synchronised regions as a **native uniform branch inside asm** (`raw_h80_vactive_native`). A PASS at one length does not revive a candidate |
| 5 | Enabling the KQ early-exit (skip) gives NaN (tiny64, clamped wholeSV) | KQ skip is unsafe in this lowering | `KQ skip` forbidden. Keep the mask and clamp every raw load. For the same reason `RAW_KQ_TAIL` is OFF |
| 6 | In a candidate that narrowed the KQ SG width with key8, SV gave NaN/Inf (1008 values, seq255) | WGK shrank 256→128, but when `causal_k-k0>128` the SV helper reads cp8+ and so reads **unwritten P-SLM** | Pass the same tile bound to producer and consumer: `SV_limit = min(WGK, causal_k-k0)` |
| 7 | Inlining the same asm block twice FAILs to compile | The vISA verifier rejects **duplicate labels** (renaming the labels showed the same symptom) | Asm containing labels appears only once per call site. Structure it so it compiles without labels (remove the tail jump) or merge the calls into one |
| 8 | `lsc_load.slm d32x128t` / `d64x64t` (512 B loads that exist in micro's native ISA) FAIL to compile: "this message accesses more than 8 registers" | **The inline asm verifier's allowed set is narrower than micro's nGEN emitter** | Being in micro's final ISA does not mean it works in inline asm. Compile the allowed messages first to confirm |
| 9 | `gather d64x8` rejected (8 GRF message limit), ExecSize 8 transposed gather (`requires 1`), strided mov stride 8 rejected (allowed 0/1/2/4) | The same verifier constraints | Keep the error logs as `*_inline_errors.txt` and do not retry the same form |
| 10 | Wrong output because a raw payload operand was not GRF aligned | A scalar raw-address payload must be GRF aligned | Build payloads with an `align=wordx32` decl or an alias |
| 11 | `dpas.w.hf...` syntax FAIL | The correct split-matrix syntax is `dpasw.hf.hf.8.8` | Check the syntax of a new instruction first in a one-line minimal kernel |
| 12 | 64 accumulator-initialisation `mov`s for the first DPAS | Zero-init done with separate movs | Using `%null.0` (`%%null.0` in a C string) as `src0` of the first depth removes the movs (142.7→140.6 µs, same accumulation order) |
| 13 | The first nativepipe gave wrong output + the `memory` clobber had no effect | #1 above (uniform declaration) | Do not use a clobber as a cure. All timings with wrong output are invalid |
| 14 | In a partial tile, *moving the origin* of the 8-row bundle with `min(k0+sg*8, k-8)` FAILs at seq 129/511/513 | Moving the origin of the last 8-row bundle also shifts the valid key positions | `k16_native_safe`: only full tiles go native; partial tiles take a different, verified path |
| 15 | At h48, the "group start clamp" of tiny V32 shifts valid channels (seq36 maxabs 3.18) | The group start clamp is safe only when head is aligned to the V tile width | h48 uses V16 tiny (4 SG, 1 padded value group) |
| 16 | `K` operand source mapping specified wrongly (first DPAS operand) | The most common mistake in hand-written asm | The initial files are discarded; only the correct files are valid. Check the DPAS operand mapping in chapter 01 first |

## 8.5 Verification protocol for asm changes

1. **Compile in minimal units**: get a new instruction/message/region through syntax and the verifier in a single-instruction kernel before putting it into a large block.
2. **Accuracy first**: NaN poison + CPU double all-rows. After passing, confirm **full-output bit-identical** against the previous OCL variant (a change that does not alter the math/accumulation order must be bit-identical). An order-changing change (key8) may differ in a few halfwords; accept that only within tolerance (always record it).
3. **Hit the boundary lengths first**: partial tile lengths (e.g. 1, 16, 31, 32, 33, 64, 65, 97, 127, 129, 255, 257, 511, 513, 1025, 1033), the last head band, odd depth (head80's 5 depths FAILed by missing the last tile at the `DKS_ACTIVE/2` floor), multiple subsequences.
4. **spill=0 / ISA diff**: changing an asm block perturbs IGC's allocation. Always check spill and the send/dpas counts.
5. **One change at a time**: do not interpret the result of a large change to the asm bundle as a single cause. Record the time of a candidate that combines many structural changes only as the "total effect".
6. **Freeze the fixed policy before measuring**: to prevent cherry-picking a candidate that happened to look good after measurement, freeze the policy/source SHA256 before measuring (chapter 09 §9.5).
7. **IGC version and allocation dependence**: physical GRF allocation and scheduling are decided by IGC. The same asm may take a different time when the IGC version changes. Review items after integration: byte pitch/operand shape of virtual register aliases, subgroup mask (`M1_NM`), barrier order, partial head/unused band/multi-iteration paths, clobber/liveness.

## 8.6 When not to use asm

- When the bottleneck is bandwidth/algorithm (e.g. an unnecessary pre-pass, a wrong tile). Fix the structure first with the root-cause procedure of chapter 06.
- When deciding on the basis of one pretty performance number. A large share of the asm/native candidates (including about 224 variants in `v/` and the `raw_stage/` candidates) were discarded for accuracy or performance.
- When the source is shared and must stay byte-identical for another architecture (SG16/Xe2). Isolate that asm inside `#if SG8` and confirm SG16 ISA identity with `python3 test/sdpa_ocl_ab.py l0 --base <commit>`.
