# CMake and Build System Review

Applies to `CMakeLists.txt`, `*.cmake`, `*.cmake.in`, `cmake/`, and toolchain
files; for dependency or CI job changes, also apply [ci.md](ci.md).

## Mandatory requirements

### General

- Follow modern CMake practices: target-scoped commands; explicit
  `PUBLIC`/`PRIVATE`/`INTERFACE`; linking to targets instead of paths; no global
  flags, includes, definitions, or warning suppressions.
- New targets use project helpers (`ov_add_target` with `SOURCES`,
  `ov_add_plugin`, `ov_add_frontend`, `ov_add_test_target`); raw
  `add_library`/`add_executable` must be justified.
- Do not collect target sources with `file(GLOB*)`, including `ROOT` and
  `ADDITIONAL_SOURCE_DIRS` of `ov_add_target`. Other globbing must be justified.
- Justify exceptions with an in-code comment and list each one in the review for
  the human approver; a missing justification is a finding.
- Internal dependencies (oneDNN, protobuf, pugixml, etc.) must not leak into the
  public interface of `openvino::runtime`.

### Options

- Avoid new public options: `ov_option*` (in `features.cmake`) adds to
  `OV_OPTIONS`, the developer package interface; use it only if out-of-tree
  plugins or extensions need it, else `option()`/`cmake_dependent_option()`.
- Defaults are correct for static builds, cross-compilation, and every OS/arch.
  A new public option defaults to the previous behavior; consumers tolerate its
  absence in an older developer package.
- Renames or removals of options need a compatibility path; document user-facing
  options in `docs/dev/cmake_options_for_custom_compilation.md`.

### Static build, install, and export

- New library targets have an `openvino::<name>` alias with matching
  `EXPORT_NAME`, plus `ov_install_static_lib` and
  `ov_developer_package_export_targets` if shipped or used elsewhere.
  Self-registering objects need whole-archive linking.
- Every `install()` sets a `COMPONENT` (new ones via `ov_cpack_add_component`)
  and uses `OV_CPACK_*DIR` destinations. Exported target names stay stable.

### CMake and OpenVINO version compatibility

- Commands, keywords, generator expressions, and policies must exist in the
  file's minimum CMake version and not be removed or deprecated in CMake 4.x.
- The build tree uses the root `cmake_minimum_required`. Files shipped
  downstream (`cmake/templates/*.cmake.in`, developer package scripts,
  `samples/`) have a lower minimum; guard newer features with `CMAKE_VERSION` or
  `if(POLICY ...)`. Raising the minimum needs a reason and CI support.
- `cmake/developer_package/**` is installed as a whole, so its functions,
  macros, variables, and exported targets are public API. Removals, renames, and
  signature changes need a deprecation path; `OpenVINOConfig-version.cmake.in`
  only compares versions and cannot detect them.

## Common problems in past PRs

- Option ON by default pulls in a target missing in static or trimmed builds.
- `PUBLIC` link to an internal library leaks it into `OpenVINOTargets.cmake`.
