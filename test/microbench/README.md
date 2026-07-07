# sdpa_ocl int8 microbenchmarks

Observer-effect-free tools for analyzing the int8 KV-cache dequant paths in
`sdpa_ocl.cl`, WITHOUT editing/building the real kernel. Two kinds:

## Static ISA microbenches (`*.cl` + `compare_isa.sh`)
Compile a small isolated .cl with ocloc for bmg, dump the GEN ISA, print instruction mix.
Use to compare reformulations by instruction/mov count.

    ./compare_isa.sh                          # all *.cl
    ./compare_isa.sh k_dequant_float.cl k_dequant_half.cl

- `k_dequant_float.cl` / `k_dequant_half.cl` — K dequant scale/zp in float vs half.
  Float wins (134/83 vs 144/93 instr/mov): GEN forces int-word->half conv to a scattered
  `<2>` dst, so half needs extra `:uw<2>` repack; float stays dense `<1>` + `{Compacted}`.
- `v_dequant_baseline.cl` — current V dequant/repack (1 cp-block, asym): ~93 mov.
- `v_dequant_A_char16` (93), `_B_nozp` (68), `_C_char2pack` (68), `_E_reinterpret` (68),
  `_F_vec_zp` (93). Lever-1 conclusion: only zp-removal (symmetric) helps; the int8->f16
  widening + per-key zp broadcast is IGC's floor. See the sdpa-ocl-int8-perf memory.

## On-device layout probes (`*.cpp` + `run_probe.sh`)
Actually run a kernel on the GPU and print the observed data layout (ground truth for
"which (key,head/value) does each lane/byte carry"). Needs libOpenCL + a GPU.

    ./run_probe.sh verify_k_transform    # K 8b-transform read -> lane==head, no shuffle
    ./run_probe.sh probe_v_layouts       # V load layouts (8b-transform is already VNNI-aligned)

## API probe
    ./probe_dpas_api.sh    # which DPAS/2d-block builtins exist on bmg (int8 yes, mixed-prec no)

## Notes
- `dump/` holds compiled ISA + built binaries; safe to delete, regenerated on demand.
- Device is hardcoded to `bmg` (Arc B580 / Battlemage / Xe2).
- Related session tools one level up: `../dump_isa.sh` (dumps the REAL sdpa_ocl vs
  sdpa_micro kernel ISA from the ov_gpu_func_tests run).
