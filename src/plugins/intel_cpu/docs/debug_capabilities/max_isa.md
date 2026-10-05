# Max ISA cap

Cap max x86 ISA at runtime. Reproduces lower-ISA behavior on higher-ISA hardware
without recompiling.

Two independent knobs:

- `OV_CPU_MAX_ISA` — dispatch based on `with_cpu_x86_*` getters.
- `ONEDNN_MAX_CPU_ISA` — oneDNN primitives, plus any dispatch based on
  `dnnl::impl::cpu::x64::mayiuse()` and CPU plugin jit kernels inheriting
  `dnnl::impl::cpu::x64::jit_generator_t` (practically almost all x64 jit kernels).

## Usage

Cap everything — set both to the same value:

```sh
OV_CPU_MAX_ISA=<isa> ONEDNN_MAX_CPU_ISA=<isa> binary ...
```

Cap one side only — narrows down which dispatch path causes a regression:

```sh
OV_CPU_MAX_ISA=AVX2 binary ...        # with_cpu_x86_* dispatch capped, oneDNN/jit unrestricted
ONEDNN_MAX_CPU_ISA=AVX2 binary ...    # oneDNN/jit capped, with_cpu_x86_* dispatch unrestricted
```

Not cross-validated — consistency is the user's responsibility. OV cannot
propagate its value to oneDNN: oneDNN caches its cap on first `mayiuse()`, and
static init order across TUs is undefined, so there is no reliable hook.

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
values on its side, so typos surface via oneDNN when both are set.

## Examples

Force the AVX2 path on AVX-512 hardware:
```sh
OV_CPU_MAX_ISA=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 ./benchmark_app -m model.xml
```

Disable AMX on Sapphire Rapids, keep AVX-512 FP16:
```sh
OV_CPU_MAX_ISA=AVX512_CORE_FP16 ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16 ./benchmark_app -m model.xml
```

## Build requirement

`OV_CPU_MAX_ISA` requires `-DENABLE_DEBUG_CAPS=ON`. In release builds the cap
check is compiled to `true` and inlined away — zero runtime cost.

`ONEDNN_MAX_CPU_ISA` is always available, since OV forces
`DNNL_ENABLE_MAX_CPU_ISA=ON` for the bundled oneDNN.

## Caveats

- `OV_CPU_MAX_ISA` read once on first ISA query. Mid-run changes ignored.
- Caps runtime detection only. Statically dispatched kernels behind `#ifdef` stay as built.
- Non-x86 builds (ARM, RISC-V) ignore `OV_CPU_MAX_ISA`.
