---
name: xe-visa-inline-asm
description: Playbook for writing and validating inline vISA assembly (__asm__ with .decl/alias, lsc_load, dpas/dpasw, predicates/labels) inside Intel Xe-HPG OpenCL kernels when IGC sinks loads or scatters packing - syntax, verified patterns, verifier limits, 16 failure modes found on DG2, and the accuracy protocol. Use when considering or reviewing native/asm in sdpa_ocl experiments.
---

# Xe inline vISA assembly

Details: `src/plugins/intel_gpu/docs/ocl_perf_guide/08-inline-visa-asm.md`. Source example: `test/sdpa_ocl_xe_hpg/s7a/perf/opt/short_diag_20261003/raw_stage/gate3_asm_first/raw_h256_qslmahead.cl` (untracked, for standalone experiments; not integrated into the product `sdpa_ocl.cl`).

## When
Only after ablation and the ISA have confirmed that the bottleneck is **scheduling/message shape/packing instructions**. (IGC sinks loads right before `dpas`, VNNI pack takes ~80 movs per operand, the same source varies by ±40%.) If the bottleneck is the algorithm·bandwidth·a pre-pass, asm does not help. The user's policy: hit the performance target first, then decide on converting to OpenCL C in the maintenance review.

## Syntax summary
```c
__asm__ volatile("{\n"
  ".decl PW v_type=G type=uw num_elts=256 alias=<%0,0>\n"     // reinterpret byte offset 0 of C variable %0 as uw
  ".decl P0 v_type=G type=ud num_elts=64 align=wordx32\n"      // GRF-aligned virtual register
  ".decl PRED v_type=P num_elts=1\n"
  "mov (M1_NM,8) PW(0,0)<2> XW(0,0)<16;8,2>\n"                 // region <vstride;width,hstride>
  "lsc_load.ugm.ca.ca (M1,8) %9:d32x8 flat[%13]:a64\n"          // global per-lane address
  "lsc_load.slm (M1_NM,1) P0:d32x64t flat[%12+0]:a32\n"         // SLM block transposed
  "dpas.hf.hf.8.8 (M1,8) %0.0 %0.0 P0.0 PD0(0,0)\n"             // for dpasw: dpasw.hf.hf.8.8
  "cmp.ne (M1_NM,1) PRED %7(0,0)<0;1,0> 0:ud\n(PRED) jmp (M1_NM,1) LBL\n" "LBL:\n"
  "}\n" : "=rw"(out) : "rw"(in) /* accumulator uses "+rw" */);
```
For the first depth, use `%null.0` (`%%null.0` in the C string) as the accumulator to remove the zero-init mov.

## What broke (to prevent recurrence)
1. A subgroup-varying value (SLM address) passed as `rw.u` (uniform) → wrong results. Varying values use `rw`. On wrong results, split the asm by function and isolate by removing pieces one at a time. A `memory` clobber does not fix it.
2. 64-bit `uq` ADD → only the low word was lowered to scalar, losing the carry (wrong results). Use `addc` or precompute in C.
3. Payload-lowering wrong results of unknown cause (h272 upper band): do not generalize; only the workaround of skipping *absent* bands with a uniform native branch was adopted.
4. Guards around barrier/SLM in OpenCL C produced NaN through compiler reordering (only at seq257). Guards are a **native uniform branch inside the asm**.
5. KQ early-exit/skip gives NaN under some lowerings → forbidden; keep the mask + clamp every raw load.
6. Mismatched producer (WGK) and consumer (SV limit) bounds → reads of unwritten SLM. `min(WGK, causal_k-k0)`.
7. Inlining asm with labels twice → verifier duplicate-label error (a case where even renaming the labels did not fix it). 1x per call.
8. **The inline verifier is narrower than micro's nGEN**: rejects `d32x128t`/`d64x64t` (512 B) "more than 8 registers", gather d64x8, ExecSize8 transposed gather, mov stride 8 (allowed 0/1/2/4). Even if it is in the micro ISA it may not work inline → compile a one-line kernel first.
9. Raw payloads are GRF-aligned (alias/`align=wordx32`). Check syntax with a minimal kernel first (`dpas.w.` is an error).
10. In a partial tile, moving the origin of the 8-row group (`min(k0+sg*8,k-8)`) also changes the valid key positions → native only for full tiles.
11. Head-alignment dependence: the group-start clamp is safe only when the head is aligned to the V tile width (h48 FAIL). Odd depth (h80, 5 depths) misses the last tile with the `/2` floor.
12. A mistake in the source mapping of the first operand is the most common → discard; only the correct file is valid.

## Patterns that helped
d64×4 gather + uq register transpose (native grouping changes the lowering, 138.6→129.3 µs), whole-KQ/whole-SV single asm, clamped whole-SV (branch removal), simplifying the V address algebra (+4.06→+1.70%), null-src0, native uniform skip (padded value group).

## Verification protocol
1. Confirm new messages/regions pass the verifier in a minimal kernel.  2. NaN poison + CPU double **all-rows**, unchanged tolerance (`maxabs<1e-2`).  3. Math-invariant changes are **full-output bit-identical** to the previous variant.  4. Boundary lengths (1,16,31–33,63–65,127–129,255–257,511–513,1025,1033)·partial band·odd depth·multiple subsequences·sharp inputs.  5. Check spill=0 and the ISA send/dpas counts.  6. One thing at a time.  7. Freeze the policy/source SHA256 before measuring.  8. Discard the timings of failures and keep only the logs.  9. Integration review: virtual register alias byte pitch/operand shape, `M1_NM` mask, barrier order, clobber/liveness, IGC version dependence. Isolate SG16/Xe2 sources with `#if SG8` and confirm byte-identity with `sdpa_ocl_ab.py l0`.
