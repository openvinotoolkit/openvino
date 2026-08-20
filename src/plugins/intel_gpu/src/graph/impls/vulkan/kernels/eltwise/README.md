# Vulkan Eltwise Shader Module

`entry.comp` is the only GLSL entry point. `program.glsl` composes the implementation in dependency order; generated
variants do not duplicate shader source.

The modules are grouped by responsibility:

- `configuration.glsl`, `bindings.glsl`, and `abi.glsl` define the compile-time contract, descriptor layouts, and
  shared host/shader constants.
- `metadata.glsl`, `storage.glsl`, and `broadcasting.glsl` own runtime metadata and tensor access.
- `integer_math.glsl` and `operations.glsl` implement Eltwise semantics.
- `evaluation*.glsl`, `post_operations.glsl`, and `fused*_evaluation.glsl` compose base, post-op, and fused evaluation.
- `dispatch*.glsl` own scalar, packed, and f32-vector execution paths.

`variants.cmake` is the complete declarative registry of host-compiled modules. Add a variant with
`add_eltwise_shader_variant(NAME ... <capabilities>)`; keep feature definitions named and grouped with the path that
owns them. The generated header name is derived from `NAME` and written only to the build tree.

Do not add one-line `.comp` wrappers or checked-in `*_spirv.hpp` files. A structural change must keep the generated
variant set and optimized SPIR-V byte-equivalent unless the change intentionally modifies shader behavior and carries
separate correctness and performance evidence.
