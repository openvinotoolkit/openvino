# CMake and Build System Review

Applies to `CMakeLists.txt`, `*.cmake`, `*.cmake.in`, `cmake/`, and toolchain
files. For dependency or CI job changes, also apply [ci.md](ci.md). Project
helpers live in `cmake/developer_package/`.

## Mandatory requirements

### General

- Follow generally accepted modern CMake practices: target-scoped commands;
  explicit `PUBLIC`/`PRIVATE`/`INTERFACE`; linking to targets instead of paths;
  no global flags, includes, definitions, or warning suppressions.
- New targets use project helpers (`ov_add_target` with `SOURCES`,
  `ov_add_plugin`, `ov_add_frontend`, `ov_add_test_target`). Raw
  `add_library`/`add_executable` must be justified.
- Do not collect target sources with `file(GLOB*)`, including the `ROOT` and
  `ADDITIONAL_SOURCE_DIRS` arguments of `ov_add_target`. Globbing for other
  purposes needs a justification.
- A required justification is an in-code comment next to the construct. List
  every justified exception explicitly in the review so the human approver
  sees it; a missing justification is a finding.
- Internal dependencies (oneDNN, protobuf, pugixml, etc.) must not leak into
  the public interface of `openvino::runtime`.

### Options

- Avoid new public options. Use `ov_option`/`ov_dependent_option`/
  `ov_option_enum` in the matching `features.cmake` only if out-of-tree
  consumers (plugins, extensions) need the option: these are added to
  `OV_OPTIONS` and become part of the developer package interface.
  Project-specific options use plain `option()` or `cmake_dependent_option()`.
- Defaults must be correct for static builds, cross-compilation, and every
  OS/arch. A new public option defaults to the previous behavior, and
  consumers must tolerate its absence when built against an older developer
  package.
- User-facing options are documented in
  `docs/dev/cmake_options_for_custom_compilation.md`. Renames or removals need a
  compatibility path.

### Static build, install, and export

- New library targets have a namespaced alias (`openvino::<name>`) with a
  matching `EXPORT_NAME`, and install/export rules (`ov_install_static_lib`,
  `ov_developer_package_export_targets`) when shipped or used by other
  components. Self-registering objects need whole-archive linking.
- Every `install()` sets a `COMPONENT` (new ones via `ov_cpack_add_component`)
  and uses `OV_CPACK_*DIR` destinations. Exported target names stay stable.

### CMake version compatibility

- Commands, keywords, generator expressions, and policies must exist in the
  minimum CMake version of the file's context and must not be removed or
  deprecated in the latest CMake (4.x).
- The build tree uses the root `cmake_minimum_required`. Files shipped to
  downstream projects (`cmake/templates/*.cmake.in`, developer package scripts,
  `samples/`) have a lower minimum; guard newer features with `CMAKE_VERSION`
  or `if(POLICY ...)`.
- Raising `cmake_minimum_required` needs a stated reason and a check that CI
  images provide that version.

### OpenVINO version compatibility

- `cmake/developer_package/**` is installed as a whole into the developer
  package, so its functions, macros, variables, and exported targets are public
  API. Removing, renaming, or changing the signature of any of them needs a
  deprecation path within the same major version.
- `OpenVINOConfig-version.cmake.in` only compares version numbers (same major
  and an older or equal requested version counts as compatible). It cannot
  detect a removed function or a renamed option, so such breaks surface only
  in downstream builds.

## Common problems in past PRs

- Option ON by default pulls in a target that is not built in a static or
  trimmed build, so install fails.
- `PUBLIC` link to an internal library leaks it into `OpenVINOTargets.cmake`.
- Static registrar dropped by the linker because whole-archive linking is
  missing.
