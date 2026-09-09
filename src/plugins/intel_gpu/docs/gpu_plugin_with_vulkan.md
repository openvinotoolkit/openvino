# GPU plugin Vulkan foundation

Vulkan is an experimental runtime of the existing `GPU` plugin. It preserves the
`openvino_intel_gpu_plugin` library name and common graph, primitives, shape
inference and model lifecycle. A build selects one runtime; a compiled model
does not mix Vulkan with OpenCL or use a hidden CPU fallback.

## Operation scope

Eltwise reuses `EltwiseKernelRef`, the existing kernel-selector JIT and
`generic_eltwise_ref.cl`. During model compilation, the plugin materializes the
OpenCL C source and calls the CLSPV library in-process. Inference consumes the
resulting SPIR-V, not an OpenCL runtime or compiler process. Serialized models
retain SPIR-V; the device-specific Vulkan pipeline cache is separate.

Reorder supplies the structural, copy and reference-kernel conversion paths
needed by Eltwise graphs; it is not general Reorder operation coverage. Reshape
remains structural. Slang supplies a build-time smoke kernel through the same
Vulkan runtime, not a production operation. Production GLSL and `glslc` are not
required. New optimized operation implementations are outside this foundation.

## Dependencies

| Dependency | Build host | Target deployment |
| --- | --- | --- |
| Vulkan headers and loader | Target SDK/sysroot | Loader and hardware driver; MoltenVK on macOS |
| CLSPV | `clspv`, `clspv-reflection` for bootstrap generation | `libclspv_core` for model compilation |
| LLVM/Clang and libclc | CLSPV's pinned dependencies, including native generators | Incorporated into the CLSPV library in the validated configuration |
| SPIRV-Tools and SPIRV-Headers | `spirv-opt`, `spirv-val`, matching headers | Target `SPIRV-Tools-static` linked into the plugin |
| Slang | `slangc` for bootstrap generation | None |

The validated CLSPV revision is
`770b077f8de4ce1018ee907f2f0a69bc41d42192`, without source modifications.
Build its shared library with `CLSPV_SHARED_LIB=ON` and
`ENABLE_CLSPV_INSTALL=ON`. `LLVM_ENABLE_ZSTD=OFF` avoids an optional compression
library dependency. Use isolated compiler launchers and `CCACHE_FOUND=FALSE`
to prevent CLSPV's automatic ccache discovery from adding a second launcher.

For cross-compilation, the CLSPV library and SPIRV-Tools archive must target the
device, while `clspv`, reflection/validation tools, Slang and LLVM/Clang table
generators must run on the build host. CLSPV accepts native generators through
`LLVM_NATIVE_TOOL_DIR`, `LLVM_TABLEGEN` and `CLANG_TABLEGEN`, and prebuilt libclc
through `CLSPV_EXTERNAL_LIBCLC_DIR`. Keep headers and libraries from matching
dependency revisions; mixing SPIRV-Tools headers from a system SDK with a newer
archive is not ABI-safe.

## Configure OpenVINO

From the OpenVINO source directory, after preparing dependencies:

```bash
cmake -S . -B build-vulkan -DENABLE_INTEL_GPU=ON \
    -DGPU_RT_TYPE=VULKAN \
    -DCLSPV_ROOT=/path/to/target/clspv/install \
    -DSPIRV-Tools_DIR=/path/to/target/spirv-tools/lib/cmake/SPIRV-Tools \
    -DSPIRV_HEADERS_INCLUDE_DIR=/path/to/matching/spirv-headers/include \
    -DCMAKE_C_COMPILER_LAUNCHER=/path/to/isolated-ccache-launcher \
    -DCMAKE_CXX_COMPILER_LAUNCHER=/path/to/isolated-ccache-launcher
```

`GPU_RT_TYPE=VULKAN` is already the default on Apple, Android and AArch64 targets,
so it can be omitted there. Provide the usual toolchain/sysroot and Vulkan SDK
settings for the target. Native Linux x86-64 requires explicit Vulkan selection.
Non-Vulkan configurations do not discover or link CLSPV, SPIRV-Tools or Slang.

Put host tools on `PATH`, or set `OV_GPU_CLSPV_EXECUTABLE`,
`OV_GPU_CLSPV_REFLECTION_EXECUTABLE`, `OV_GPU_SLANGC_EXECUTABLE`,
`OV_GPU_SPIRV_OPT_EXECUTABLE` and `OV_GPU_SPIRV_VAL_EXECUTABLE`. If using another
compiler revision/configuration, update `CLSPV_COMPILER_ID` to distinguish its
kernel cache entries. Build `openvino_intel_gpu_plugin` with the configured
cached build workflow; bootstrap artifacts are generated incrementally in the
build tree.

For Android, follow the [OpenVINO Android build guide](../../../../docs/dev/build_android.md).
Build OneTBB with the same NDK, ABI, Android platform and shared STL; configure
OpenVINO with `THREADING=TBB` and its `TBB_DIR`. Deploy the matching OneTBB and
`libc++_shared.so` libraries. Pi 4 and Pi 5 can share one Linux AArch64 package,
but require independent hardware validation.

## Runtime profile and packaging

The foundation requires Vulkan 1.3 and `storageBuffer8BitAccess`, together with
the synchronization and maintenance features checked by the Vulkan device
profile. Supported storage types are not equivalent to native arithmetic
features. Eltwise model lowering uses the common GPU type policy; native direct
I64 kernels remain unavailable on devices without `shaderInt64`. Int8/Float16
coverage must not be inferred solely from the reported arithmetic feature bits.

The normal OpenVINO runtime install component places `libclspv_core` beside the
plugin. Its LLVM/Clang payload is substantial: measured development builds add
approximately 90–123 MiB for this library alone, excluding OpenVINO and the
Vulkan driver. Keep model-compilation cost separate from inference performance.

Before redistributing a package, inspect transitive dependencies with `otool -L`
or `readelf -d`, verify loader search paths from the installed location, and
include applicable upstream third-party notices. The build-tree copy is not a
claim of a self-contained redistributable SDK; in particular, macOS development
builds can still reference an absolute Vulkan-loader path.

On macOS, perform the usual package signing after install-time RPATH changes.
Those changes invalidate existing ad-hoc signatures. Local development copies
can be ad-hoc signed again; that is not Developer ID signing or notarization
for distribution. Do not disable platform security checks to run the package.
