# GPU dual-runtime dispatch (COMBINED build)

The GPU plugin can be built for a single compute runtime (OpenCL or Level Zero), selected
at compile time by `GPU_RT_TYPE`. It can also be built in a **COMBINED** mode that ships
**both** runtime libraries side by side under the single public device name `GPU` and picks
the right one **per physical device at runtime**.

This is an opt-in build. **The default build is unchanged**: without `-DGPU_RT_TYPE`, the
plugin is built single-runtime OpenCL as before, producing one library
`openvino_intel_gpu_plugin`.

## How to build

```sh
cmake -DGPU_RT_TYPE=COMBINED -DBUILD_SHARED_LIBS=ON <other options> ..
cmake --build . --config Release
```

`COMBINED` requires `BUILD_SHARED_LIBS=ON` (it ships two plugin libraries via `plugins.xml`);
a static build with `COMBINED` fails fast at configure time. Building it produces two
libraries:

- `openvino_intel_gpu_plugin_ocl` — the OpenCL (OCL) build (listed first);
- `openvino_intel_gpu_plugin` — the Level Zero (ZE) build.

Both are shipped in the `gpu` package/component, and `plugins.xml` registers them under one
device name, in that order:

```xml
<plugin name="GPU">
    <location>libopenvino_intel_gpu_plugin_ocl.so</location>
    <location>libopenvino_intel_gpu_plugin.so</location>
</plugin>
```

The order matters for **numbering**: `GPU.0`, `GPU.1`, … are assigned in the order the
libraries enumerate devices, first library first. OCL is listed first so the numbering matches
a single-runtime OCL build for every device OpenCL can see, and a `COMBINED` install does not
renumber the GPUs your scripts already address. Devices only Level Zero enumerates are appended
after. Order does not affect *which* runtime is chosen — that is decided per device by score,
and no two scores can tie.

Single-runtime builds (`-DGPU_RT_TYPE=OCL` / `ZE` / `SYCL`) are unaffected and keep the
single-`location` registration.

## How a runtime is chosen per device

`ov::Core` asks each library which devices it can serve and how well, via a lightweight
enumeration probe (no engine is constructed), then picks the best-scoring library per device:

- **Level Zero** is preferred on **Xe2 and newer** GPUs that support ZE↔OCL interop
  (`supports_leo`) — the better runtime there, without breaking OpenCL-interop applications.
- **OpenCL** serves everything else: pre-Xe2 Intel GPUs (better runtime there), Xe2+ GPUs
  whose driver lacks interop, and all non-Intel GPUs. Level Zero is never selected for a
  non-Intel GPU.

Different physical GPUs in one system can therefore resolve to different runtimes (e.g.
`GPU.0` → ZE, `GPU.1` → OCL); both plugin instances coexist in the process.

## Forcing a runtime for debugging

Set `OV_GPU_RUNTIME=OCL` or `OV_GPU_RUNTIME=ZE` to force a runtime for Intel GPUs, bypassing
the automatic selection. It is a debugging aid only (no public config property). An
unrecognized value is ignored, and it can never attach Level Zero to a non-Intel GPU.

An Intel GPU that the forced runtime cannot serve becomes **unavailable** while the variable is
set: it is dropped from `available_devices` and addressing it reports that no library can serve
it. Its `GPU.N` slot is still reserved, so the override never renumbers the other GPUs.

## On-disk cache

The compile-time runtime tag (`OCL`/`ZE`/`SYCL`) is part of the key of every on-disk GPU
cache (model blob, kernel `.cl_cache`, oneDNN `.onednn.cl_cache`), so OpenCL and Level Zero
never share a cache file and a binary compiled by one runtime is never loaded by the other.
Both runtimes' caches persist in `cache_dir` and are independently reusable.
