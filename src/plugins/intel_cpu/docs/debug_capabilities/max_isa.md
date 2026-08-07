# Max ISA cap

Cap the maximum x86 ISA used at runtime for both OV CPU kernels and oneDNN.
Useful to reproduce lower-ISA behavior on higher-ISA hardware (e.g., force
AVX2 path on an AVX-512 machine) without recompiling.

## Usage

Set **both** variables to the same value:

```sh
OV_CPU_MAX_ISA=<isa> ONEDNN_MAX_CPU_ISA=<isa> binary ...
```

- `OV_CPU_MAX_ISA` caps ISA dispatch inside OV CPU kernels (`with_cpu_x86_*` getters).
- `ONEDNN_MAX_CPU_ISA` caps ISA dispatch inside oneDNN primitives.

Both must be set to the **same value**. On mismatch OV throws at first ISA
query. Reason: oneDNN caches its cap on first `mayiuse()` call and static
init order across TUs is undefined, so library-side propagation is not
reliable. Requiring both eliminates skew.

## Supported values

Case-insensitive. Ordered from lowest to highest:

- `SSE41` (alias: `SSE42`)
- `AVX`
- `AVX2`
- `AVX2_VNNI`
- `AVX2_VNNI_2`
- `AVX512_CORE`
- `AVX512_CORE_VNNI`
- `AVX512_CORE_BF16`
- `AVX512_CORE_FP16`
- `AVX512_CORE_AMX`
- `AVX512_CORE_AMX_FP16`
- `ALL` / `DEFAULT` / unset — no cap

Unknown values are treated as no cap on the OV side. oneDNN rejects unknown
values on its side, so typos surface via oneDNN.

## Examples

Force AVX2 path on AVX-512 hardware:
```sh
OV_CPU_MAX_ISA=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 ./benchmark_app -m model.xml
```

Disable AMX on Sapphire Rapids, keep AVX-512 FP16:
```sh
OV_CPU_MAX_ISA=AVX512_CORE_FP16 ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16 ./benchmark_app -m model.xml
```

