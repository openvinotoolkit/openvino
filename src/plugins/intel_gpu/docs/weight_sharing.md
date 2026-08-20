# Weight Sharing

This document explains what weight sharing is in the Intel GPU plugin, when it is used, and how to activate it. The implementation is exercised by [src/plugins/intel_gpu/tests/functional/behavior/weight_context.cpp](../tests/functional/behavior/weight_context.cpp) and is consumed by the constant-import path in [src/plugins/intel_gpu/src/plugin/ops/constant.cpp](../src/plugin/ops/constant.cpp#L104-L146).

## Overview

Weight sharing allows the GPU plugin to reuse a single backing buffer for constants that are logically shared across model instances compiled by different plugins (NPUW is supported at present). Instead of allocating a separate copy for each constant, the plugin can import a shared host buffer and treat it as the source for a constant.

In the GPU plugin, the imported constant path checks whether a constant belongs to a shared source buffer by performing a lookup in the weight-sharing metadata (`ov::weight_sharing::Context`). If it exists, the plugin registers the source buffer as a retained dependency and imports the constant data by using that shared buffer instead of creating a fresh internal copy in device memory.

This is the path used in [src/plugins/intel_gpu/src/plugin/ops/constant.cpp](../src/plugin/ops/constant.cpp#L104-L146):

- the constant checks whether a weight-sharing context is present,
- it reads the constant source ID and source buffer,
- it keeps the source buffer alive via `ProgramBuilder::m_shared_weight_sources`,
- and it imports the shared memory with `create_hostbuffer(...)` when the backing buffer is valid.

## Why it matters

Weight sharing is useful when multiple models or compiled models need to share the same constant payload without duplicating memory. It can reduce memory overhead and improve the reuse of constant data.

The design intentionally separates two things:

- the logical weight-sharing metadata (`ov::weight_sharing::Context`), and
- the actual backing storage (`SharedBuffer` / `AlignedBuffer`).

The plugin keeps the backing storage alive as long as the compiled graph can still reference the imported constant.

## How to activate weight sharing

The feature is activated by passing both a path to a model-caching storage and an internal runtime property when compiling the model.

### Minimal pattern

```cpp
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/internal_properties.hpp"

ov::Core core;
const ov::internal::WeightSharingCtxPtr shared_ctx = ...;
ov::AnyMap properties = {
    {ov::cache_dir("/tmp/weight-sharing-cache.bin")},
    {ov::internal::model_sharing_context(shared_ctx)}
};
auto compiled = core.compile_model(model, "GPU", properties);
```

## The two required pieces

### 1. `ov::cache_dir`

First, we need to activate the cache feature by using the property `ov::cache_dir`:

```cpp
properties = ov::AnyMap{
    {ov::cache_dir(cache_file_path)}
};
```

The `cache_file_path` must point to a single file with the `.bin` extension. In the current implementation, single-file caching automatically activates this weight-sharing flow. It becomes part of the compile configuration that prepares the model compilation environment and provides a cache path for the GPU plugin.

### 2. `ov::internal::model_sharing_context`

This is the actual activation property for the weight-sharing feature. It carries the `ov::weight_sharing::Context` object to the GPU plugin during compilation.

```cpp
properties = ov::AnyMap{
    {ov::cache_dir(cache_file_path)},
    {ov::internal::model_sharing_context(shared_ctx)}
};
```

Here, `shared_ctx` is an instance of `ov::internal::WeightSharingCtxPtr`, which must be populated by a shared-memory provider and hold information about constants according to the `ov::weight_sharing::Context` API.

The plugin extracts this property from the compile config and propagates the context into `ProgramBuilder` so it can use the constant metadata during model compilation and reuse memory previously allocated by the provider.

## Required model setup

The shared-weights provider must produce the source buffer and registered constants along with the corresponding `ov::weight_sharing::Context`. In the current implementation, this shared-weights provider is NPUW. It preprocesses a model with constants, collects the constants, and then constructs a shared source buffer with a deterministic alignment layout.

A typical setup looks like this:

- create a model with constant nodes,
- collect the constant operations,
- create a `SharedBuffer` over a source allocation,
- register the source buffer and constants in `ov::weight_sharing::Context`,
- pass the resulting context through `ov::internal::model_sharing_context` in the compile properties.

## Alignment constraint and validation

To guarantee that memory for constants allocated by a shared-weights provider can be imported by the GPU, the memory must be properly aligned according to the device capabilities. In the current implementation, all memory chunks must be page-aligned, or an invalid imported host buffer will fail during GPU memory import. In other words, the shared buffer must satisfy the backend import requirements, especially the alignment and lifetime constraints, or the plugin will reject it.

## Practical guidance

For a user or developer working with the GPU plugin:

1. Build an `ov::weight_sharing::Context` for the constants that should be shared.
2. Provide a valid cache path in `ov::cache_dir`.
3. Provide the context through `ov::internal::model_sharing_context` in the compile config.
4. Ensure the backing buffer remains alive for the full lifetime of the compiled model and all imported constants.
5. Respect the backend alignment requirements for imported host memory; otherwise, compilation may fail.

## Summary

The weight-sharing feature is activated by combining:

- `ov::cache_dir(...)` in the compile config, and
- `ov::internal::model_sharing_context` with a valid `ov::weight_sharing::Context`.

Once enabled, the GPU plugin recognizes shared constant sources, retains the underlying buffer, and imports them via host-backed OpenCL memory. The behavior is validated by the weight-sharing test and is enforced by the constant-import path in the GPU plugin.