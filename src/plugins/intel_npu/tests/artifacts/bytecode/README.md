<!--
Copyright (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Bytecode Compatibility Artifacts

This directory holds the bytecode compatibility artifacts used by the virtual-machine backward-compatibility tests. Each artifact is a small, hand-crafted bytecode binary produced from a YAML config, then embedded into the unit tests to verify that the current Virtual Machine still parses, loads and executes previously released bytecode.

The binary artifacts are frozen after generation, as they serve as golden references for validating the backward compatibility of the Virtual Machine.

> Note: The testing infrastructure for running the binaries makes use of the unit test framework. This infrastructure does not go through the NPU plugin, and instead relies on the Virtual Machine API directly. As a result, there are no Level-Zero objects available for the execution, so kernel submission and buffer management cannot be validated from the functionality point of view. The binary artifacts that make use of instructions for kernel submission and buffers are only validated at the parsing stage.

```
configs/                          YAML configs describing each artifact
binary/                           Generated .bin files (committed, consumed by tests)
generate_bytecode_artifact.py     CLI entry point
bytecode_gen/                     Generator package
  encoders.py                     little-endian byte encoders
  model.py                        section/opcode/type enums and dataclasses
  function_builder.py             per-function instruction stream builder
  artifact_builder.py             binary artifact assembly
  config.py                       YAML parsing and config-driven generation
```

## Generating artifacts

Install dependencies first:

```bash
python3 -m pip install -r tests/artifacts/bytecode/requirements.txt
```

Run the script with one or more configs and an output directory:

```bash
python3 tests/artifacts/bytecode/generate_bytecode_artifact.py \
    tests/artifacts/bytecode/configs/*.yaml \
    --output-dir tests/artifacts/bytecode/binary/
```

The output filename comes from the config's `output` field (falling back to the config filename with a `.bin` suffix).

## Config file format

A config is a YAML mapping. Sections are emitted in a fixed order that mirrors the bytecode writer; declaring a section as an empty list forces an empty section to be emitted (useful for crafting malformed artifacts).

### Top-level keys

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `output` | str | — | Output binary filename (prefixed by `--output-dir` path). |
| `version` | [int, int, int] | `[1, 0, 0]` | Format version `[major, minor, patch]`. |
| `magic_number` | str / bytes | `NPUByte\0` | File magic; override to fabricate an invalid header. |
| `allow_zero_sections` | bool | `false` | Permit an artifact with no sections. |
| `truncate_bytes_from_end` | dict | — | Truncate the binary during generation. |
| `type_section` | list | `[]` | Type entries (see below). |
| `constant_section` | list | `[]` | Raw payload entries. |
| `kernel_section` | list | `[]` | Raw payload entries. |
| `string_section` | list[str] | auto | Explicit string table (normally auto-interned). |
| `metadata_section` | list | `[]` | Network/IO metadata entries. |
| `function_section` | list | `[]` | Function definitions. |
| `unknown_sections` | list | `[]` | Opaque sections with arbitrary type bytes. |
| `section_header_overrides` | list | `[]` | Patch emitted section headers. |

A **raw payload** (constant/kernel entries, `unknown_sections.data`) is either a list of byte values `[0, 1, 255]`, a `0x`-prefixed hex string `"0x0a1b"`, or plain UTF-8 text.

### `type_section`

Each entry has an `id` (referenced by name elsewhere) and a `kind`:

```yaml
type_section:
  - id: i64
    kind: integer        # width (int), is_signed (bool)
    width: 64
    is_signed: true
  - id: f32
    kind: float          # width (int), format (default ieee754)
    width: 32
    format: ieee754      # ieee754 | bfloat | tfloat | e4m3 | e5m2 | e2m1 | e8m0 | nf4
  - id: fn_main
    kind: function       # params/results are type refs (id strings or int indices)
    params: []
    results: [i64, i64]
```

### `function_section`

```yaml
function_section:
  - name: main_int_arith     # function name, can be referenced by the test if entry point
    type: fn_main            # function type
    num_registers: 8         # optional; inferred from used registers otherwise
    instructions:
      - op: set_imm
        args: [0, 10]        # reg 0 = immediate 10
      - op: add_i64
        args: [3, 0, 1]      # reg 3 = reg 0 + reg 1
      - op: retv
        args: [3]
```

Instructions accept three forms:

- `op: <helper>` with positional `args: [...]` — e.g. `add_i64`, `ret`, `jmp`, `set_imm`, `retv`, `call`. Fixed-arity arithmetic/convert opcodes are named after the opcode (lowercased), e.g. `ASSERT` is spelled `assert`.
- `op: <helper>` with named arguments for variable-shape ops, e.g. `buffer_create` / `buffer_subview` / `kernel_create`:
  ```yaml
  - op: buffer_create
    dst: 0
    elem_type: 0
    shape_regs: [3]
    stride_regs: [4]
  ```
- Escape hatch: `op: instruction` with `opcode: <ENUM_NAME>` and `args: [...]`, or `op: <int opcode>` with `args: [...]` for raw/invalid opcodes.

Register operands are non-negative integer indices; the highest index used sets the register count unless `num_registers` is given.

### `metadata_section`

```yaml
metadata_section:
  - kind: network            # name, num_streams (default 1), num_cmdlists (default 1)
    name: compat_network
  - kind: input              # also: output, profiling_output
    name: input0
    precision_type: f32      # type ref
    shape: 0                 # constant-section index
    index_used_by_driver: 0  # uint32
    has_dynamic_strides: false
    # optional: output_tensor_names (list[str], <=255),
    #           shape_from_ir_model (constant index), node_friendly_name (str)
```

### `section_header_overrides` and `unknown_sections`

Used to fabricate malformed artifacts:

```yaml
section_header_overrides:
  - section_type: function         # alias (function|constant|string|kernel|type|metadata) or int byte
    section_occurrence: 0          # which matching section (default 0)
    offset: 18446744073709551500   # also: name_index, size
unknown_sections:
  - section_type: 0x7F             # type byte not in the current SectionType enum
    data: [0x01, 0x02, 0x03]
```

## Use in the testing infrastructure

The committed `binary/*.bin` files are embedded into the Virtual Machine unit tests at build time:

1. `tests/unit/CMakeLists.txt` calls `npu_embed_bytecode_artifacts_header`, which expands `tests/unit/virtual_machine/compatibility/bytecode_artifacts.hpp.inc` into a generated `bytecode_artifacts.hpp`, inlining each artifact's raw bytes.
    - Artifact files are resolved from placeholder names in the template: for `@ARTIFACT_<NAME>_BYTES@`, CMake maps to `tests/artifacts/bytecode/binary/<lowercase(NAME)>.bin`. Example: `@ARTIFACT_BYTECODE_V1_0_0_ALL_SECTIONS_BYTES@` maps to `tests/artifacts/bytecode/binary/bytecode_v1_0_0_all_sections.bin`. If a file is not found, a build warning is printed and empty artifact data is emitted into the generated header (size `0` and empty bytes), which causes the corresponding unit test to fail.
2. The template registers every artifact in a `BYTECODE_ARTIFACTS` map of `ArtifactMetadata { testType, entryPointFuncName, inputValues, expectedValues }`.
   `TestType` is one of:
   - `Accurate` — parse, load, run the entry point and compare outputs.
   - `ParseSuccess` — must parse and load.
   - `ParseFailure` — must fail to parse (malformed artifacts).
   - `RuntimeFailure` — parses, but execution must fail.
3. `backward_compatibility.cpp` (`VirtualMachineBackwardCompatibilityTest`) is parameterized over the map and exercises each artifact through the `npu_vm_*` API.

### Adding a new artifact

1. Add a YAML config under `configs/` and generate its `.bin` into `binary/`.
2. Add a matching `std::array` placeholder line and a `BYTECODE_ARTIFACTS` entry (with `testType`, entry-point name, inputs and expected outputs) in `bytecode_artifacts.hpp.inc`.
