# Packaging: Adding Components and Building Custom Archives

This document explains how OpenVINO composes distribution archives from CPack
components and how **any user** can:

- add a new installable component to the build;
- add, exclude, or modify the content of an archive;
- build a **custom OpenVINO package** from an arbitrary set of components.

The mechanism is the same everywhere — local developer builds, downstream
projects, and automated (CI/CD) release flows all rely on CMake/CPack
components. Nothing here depends on any internal infrastructure.

## Table of Contents

- [Concepts](#concepts)
- [The CPack component model in OpenVINO](#the-cpack-component-model-in-openvino)
  - [How a component is declared](#how-a-component-is-declared)
  - [Default `ALL` target vs. non-default components](#default-all-target-vs-non-default-components)
  - [Per-generator include/exclude rules](#per-generator-includeexclude-rules)
  - [Excluding a component by disabling it at configure time](#excluding-a-component-by-disabling-it-at-configure-time)
- [Adding a new component](#adding-a-new-component)
  - [Grouping several targets into one component](#grouping-several-targets-into-one-component)
- [How the default archive is composed](#how-the-default-archive-is-composed)
- [Building a custom archive from selected components](#building-a-custom-archive-from-selected-components)
  - [Option A — one archive from a selected component set (recommended)](#option-a--one-archive-from-a-selected-component-set-recommended)
  - [Option B — per-component staging with `cmake --install`](#option-b--per-component-staging-with-cmake---install)
  - [Common archive composition cases](#common-archive-composition-cases)
- [How documentation and licenses are added](#how-documentation-and-licenses-are-added)
- [Automating custom archives](#automating-custom-archives)
  - [Manifest format](#manifest-format)
  - [Optional CMake helper `ov_cpack_add_archive`](#optional-cmake-helper-ov_cpack_add_archive)
- [Reference: public component names](#reference-public-component-names)

## Concepts

OpenVINO uses [CPack components](https://cmake.org/cmake/help/latest/module/CPackComponent.html).
Every installable artifact (a library, a plugin, headers, samples, a license
file, a doc file, ...) is attached to a **component** via an `install(...
COMPONENT <name> ...)` command. An archive is a **set of components** packed
together.

There are two orthogonal properties for every component:

1. **Membership in the default `ALL` install target** — whether the component is
   installed by a plain `cmake --install` / packed into the default public
   archive. Controlled by the `EXCLUDE_FROM_ALL` marker.
2. **Which generator is active** — `TGZ`/`ZIP` (archives), `DEB`, `RPM`, `NPM`,
   `CONDA-FORGE`, etc. Each generator can re-map, merge, or exclude components.

A **custom archive** is nothing more than "pack this explicit list of
components", independently of what the default `ALL` target contains. This is
the core flexibility of the packaging system: any combination of components can
be turned into a standalone package.

## The CPack component model in OpenVINO

### How a component is declared

Components are registered with the wrapper `ov_cpack_add_component`
(see [`cmake/developer_package/packaging/packaging.cmake`](../../cmake/developer_package/packaging/packaging.cmake)),
then artifacts are attached with `install(... COMPONENT ... )`.

Example — the C++ samples component
([`samples/CMakeLists.txt`](../../samples/CMakeLists.txt)):

```cmake
ov_cpack_add_component(${OV_CPACK_COMP_CPP_SAMPLES}
                       HIDDEN
                       DEPENDS ${OV_CPACK_COMP_CORE_DEV})

install(DIRECTORY cpp/
        DESTINATION ${OV_CPACK_SAMPLESDIR}/cpp
        COMPONENT ${OV_CPACK_COMP_CPP_SAMPLES}
        ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL}   # <- membership marker
        PATTERN *.sh EXCLUDE)
```

Key parts:

- `ov_cpack_add_component(<name> ...)` — registers the component and records its
  `cpack_add_component` arguments so they survive across the `cpack` invocation.
- `COMPONENT ${...}` — attaches installed files to the component.
- `${OV_CPACK_COMP_<NAME>_EXCLUDE_ALL}` — expands to either nothing (in `ALL`)
  or `EXCLUDE_FROM_ALL` (not in `ALL`), depending on the active generator.

### Default `ALL` target vs. non-default components

Whether a component is part of the default install is decided by the value of
the `OV_CPACK_COMP_<NAME>_EXCLUDE_ALL` variable, set per generator in
[`cmake/developer_package/packaging/archive.cmake`](../../cmake/developer_package/packaging/archive.cmake)
(`ov_define_component_include_rules`):

```cmake
# part of the default public archive -> variable is empty
unset(OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL)

# NOT part of the default archive -> variable == EXCLUDE_FROM_ALL
set(OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
set(OV_CPACK_COMP_NPM_EXCLUDE_ALL EXCLUDE_FROM_ALL)
```

- **Empty value** → the component is installed by `cmake --install` and is part
  of the default archive.
- **`EXCLUDE_FROM_ALL`** → the component still exists and can be packed
  explicitly by name, but it is **not** in the default archive. This is how a
  component can be kept out of the default `ALL` target for the platform, yet
  still be requested for a custom archive.

Whether a given component is in `ALL` is **generator-dependent**: the same
component may be in `ALL` for archives (`TGZ/ZIP`) but excluded for `DEB/RPM/NPM`.
For the archive generator, components marked non-`ALL` include `python_wheels`
and `ov_node_addon` (NPM), see
[`archive.cmake`](../../cmake/developer_package/packaging/archive.cmake).

Any component — regardless of its default-`ALL` membership — can be installed
independently by name, which is the basis for building custom archives:

```bash
# npu_internal is in ALL for archives, but can still be installed on its own:
cmake --install <build_dir> --component npu_internal --prefix <stage_dir>
```

The `npu_internal` component (e.g. `compile_tool`) is attached like any other
component, see
[`src/plugins/intel_npu/tools/compile_tool/CMakeLists.txt`](../../src/plugins/intel_npu/tools/compile_tool/CMakeLists.txt);
its default-`ALL` membership is controlled per generator via
`OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL`:

```cmake
install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION "tools/${TARGET_NAME}"
        COMPONENT ${NPU_INTERNAL_COMPONENT}
        ${OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL})
```

### Per-generator include/exclude rules

Each generator file under
[`cmake/developer_package/packaging/`](../../cmake/developer_package/packaging/)
defines its own `ov_define_component_include_rules()` and directory layout
(`ov_<generator>_cpack_set_dirs()`). The archive generator
(`TGZ/ZIP/...`) rules live in `archive.cmake`. This is where you change whether
a stock component belongs to the default archive on a given platform.

### Excluding a component by disabling it at configure time

Many components are produced only when their build feature is enabled. Turning
the corresponding `ENABLE_*` option off at CMake configuration time removes the
component (and its files) from the build entirely, so it is absent from every
archive without touching any packaging rule. For example:

```bash
# Drop the NPU plugin and its components from the package:
cmake -DENABLE_INTEL_NPU=OFF <source_dir>

# Drop Python bindings / wheels:
cmake -DENABLE_PYTHON=OFF <source_dir>

# Drop C++/C samples:
cmake -DENABLE_SAMPLES=OFF <source_dir>
```

Use this when a component should never be built for a given configuration.
Use the `EXCLUDE_FROM_ALL` marker (above) instead when the component must still
be built and be available for explicit, on-demand packaging.

## Adding a new component

To ship a new artifact (tool, library, extra file set), register a component and
attach files to it. The component then becomes selectable for any archive.

1. **Register the component name.** Reuse an existing
   `OV_CPACK_COMP_*` name, or add your own next to
   `ov_define_component_names()` in
   [`cmake/developer_package/packaging/packaging.cmake`](../../cmake/developer_package/packaging/packaging.cmake).

2. **Declare the component and (optionally) its default-`ALL` membership.**
   In your `CMakeLists.txt`:

   ```cmake
   # Register the component (records cpack_add_component args across cpack).
   ov_cpack_add_component(my_tool HIDDEN DEPENDS ${OV_CPACK_COMP_CORE})

   # Attach files to it.
   install(TARGETS my_tool
           RUNTIME DESTINATION tools/my_tool
           COMPONENT my_tool
           ${OV_CPACK_COMP_MY_TOOL_EXCLUDE_ALL})   # membership marker (optional)
   ```

3. **Decide default-`ALL` membership** by setting the marker variable in the
   generator rules (`ov_define_component_include_rules` in
   [`archive.cmake`](../../cmake/developer_package/packaging/archive.cmake)):

   ```cmake
   # In the default archive:
   unset(OV_CPACK_COMP_MY_TOOL_EXCLUDE_ALL)
   # OR available only on explicit request (custom archives):
   set(OV_CPACK_COMP_MY_TOOL_EXCLUDE_ALL EXCLUDE_FROM_ALL)
   ```

   If you do not define the marker variable at all, the `${...}` expands to
   nothing and the component is part of `ALL`.

4. **Pack it** — either it flows into the default archive automatically (if in
   `ALL`), or you list it explicitly when building a custom archive (see
   [below](#building-a-custom-archive-from-selected-components)).

### Grouping several targets into one component

A single component can gather multiple targets and files, so that the whole
group is installed by **one** `cmake --install --component` command (and packed
by listing a single component name). Assign the same `COMPONENT` to every
`install(...)` call:

```cmake
# One component that bundles a plugin, a helper library and a CLI tool.
ov_cpack_add_component(my_bundle HIDDEN DEPENDS ${OV_CPACK_COMP_CORE})

install(TARGETS my_plugin
        LIBRARY DESTINATION ${OV_CPACK_PLUGINSDIR}
        COMPONENT my_bundle
        ${OV_CPACK_COMP_MY_BUNDLE_EXCLUDE_ALL})

install(TARGETS my_helper
        LIBRARY DESTINATION ${OV_CPACK_LIBRARYDIR}
        ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR}
        COMPONENT my_bundle
        ${OV_CPACK_COMP_MY_BUNDLE_EXCLUDE_ALL})

install(TARGETS my_cli
        RUNTIME DESTINATION tools/my_cli
        COMPONENT my_bundle
        ${OV_CPACK_COMP_MY_BUNDLE_EXCLUDE_ALL})

# Extra non-target files can join the same component too.
install(FILES README.md
        DESTINATION tools/my_cli
        COMPONENT my_bundle
        ${OV_CPACK_COMP_MY_BUNDLE_EXCLUDE_ALL})
```

All targets and files above are now installable and packable as one unit:

```bash
# Installs my_plugin + my_helper + my_cli + README.md in a single command.
cmake --install <build_dir> --component my_bundle --prefix <stage_dir>
```

`install(TARGETS a b c ... COMPONENT my_bundle)` — listing multiple targets in a
single `install` call — is also valid when they share the same destination
kind, and behaves identically at pack time.

## How the default archive is composed

At the end of configuration `ov_cpack(<components...>)` is called
(`cmake/developer_package/packaging/packaging.cmake`). It:

1. Sets `CPACK_COMPONENTS_ALL` to the passed component list.
2. Calls the generator-specific `ov_cpack_settings()` (e.g. in
   [`cmake/packaging/archive.cmake`](../../cmake/packaging/archive.cmake)),
   which filters the list (removes components marked `EXCLUDE_ALL`, wheels,
   `jax`, etc.).
3. Calls `include(CPack)`.

For archives, the relevant CPack switch is set in `archive.cmake`:

```cmake
# ON  -> one archive per component
# OFF -> a single archive that contains all listed components
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
```

The default archive for a platform is therefore "the set of components that are
in `ALL` for the archive generator on that platform".

## Building a custom archive from selected components

There are two supported ways to produce an archive whose content differs from
the default. Both operate on **CPack components**, never on the CMake `ALL`
build target.

### Option A — one archive from a selected component set (recommended)

Override `CPACK_COMPONENTS_ALL` with the exact component list and group them
into a single archive:

```bash
cmake -S <source_dir> -B <build_dir> \
      -DCPACK_GENERATOR=TGZ \
      -DCPACK_ARCHIVE_COMPONENT_INSTALL=OFF

cpack --config <build_dir>/CPackConfig.cmake \
      -D CPACK_COMPONENTS_ALL="core;core_c;cpp_samples" \
      -D CPACK_ARCHIVE_FILE_NAME="openvino_custom_runtime" \
      -B <output_dir>
```

- `CPACK_COMPONENTS_ALL` — the explicit component list for the archive. Any
  component may be listed, including ones marked `EXCLUDE_FROM_ALL`, because
  CPack packs by component name regardless of the default `ALL` membership.
- `CPACK_ARCHIVE_COMPONENT_INSTALL=OFF` — merge listed components into a
  single archive instead of one-per-component.
- `CPACK_ARCHIVE_FILE_NAME` — the custom archive name.

### Option B — per-component staging with `cmake --install`

Stage each component into a directory, then compress. Useful when the archive
must mix components and extra files (docs, licenses) with fine-grained control:

```bash
for comp in core core_c licensing; do
  cmake --install <build_dir> --component "$comp" --prefix <stage_dir>
done
tar -czvf openvino_custom_runtime.tar.gz -C <stage_dir> .
```

This is essentially what the release jobs do to build the main TGZ archive;
custom archives differ only in the component list.

### Common archive composition cases

The mechanism above covers every archive composition case:

| Case | Meaning | How to produce |
|---|---|---|
| Subset | A **subset** of the default components | List only the wanted subset in `CPACK_COMPONENTS_ALL` |
| Recombination | A **different combination** of default components | List any combination of default components |
| Non-default | Components **not** in the default `ALL` for the platform | List the `EXCLUDE_FROM_ALL` component name explicitly (e.g. `python_wheels`, `ov_node_addon`) |

No CMake `ALL`-target rebuild is required for any of these — only the CPack
component selection changes.

## How documentation and licenses are added

Documentation and license files are ordinary CPack components. They are attached
to dedicated components that can be included or excluded per generator/platform,
exactly like any other component.

- **Licenses** — the `licensing` component
  ([`licensing/CMakeLists.txt`](../../licensing/CMakeLists.txt)):

  ```cmake
  ov_cpack_add_component(${OV_CPACK_COMP_LICENSING} HIDDEN)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
          DESTINATION ${OV_CPACK_LICENSESDIR}          # -> licenses/
          COMPONENT ${OV_CPACK_COMP_LICENSING}
          ${OV_CPACK_COMP_LICENSING_EXCLUDE_ALL}
          PATTERN CMakeLists.txt EXCLUDE)
  ```

- **Documentation / third-party notices** are installed into
  `${OV_CPACK_DOCDIR}` (`docs/`) and `${OV_CPACK_LICENSESDIR}` (`licenses/`)
  using the same `install(... COMPONENT ... ${..._EXCLUDE_ALL})` pattern.

To add license/doc files to a custom archive:

1. Attach the files to a component (reuse `licensing` or add a new component
   with `ov_cpack_add_component`).
2. Include that component name in the archive's `CPACK_COMPONENTS_ALL`
   (Option A) or stage it with `cmake --install --component` (Option B).

Because these are regular components, you can add previously non-public
docs/licenses to any custom archive purely by listing the component, without
patching the archive contents by hand.

## Automating custom archives

To keep custom archive definitions maintainable (comparable to a JSON manifest),
declare them **as data**, then materialize them with a small driver script or
CI step alongside the main archive.

### Manifest format

Define each custom archive with:

- `name` — custom archive file name;
- `components` — list of CPack components to include;
- `exclude` *(optional, discussible)* — components to remove from the default
  `ALL` set when the archive is derived from `ALL`;
- `platforms` — OS/arch where the archive is generated;
- `upload` — whether to publish it alongside the main public archive.

Example `packaging/custom_archives.json`:

```json
[
  {
    "name": "openvino_runtime_min",
    "components": ["core", "core_c", "tbb", "licensing"],
    "platforms": ["linux_ubuntu_22_04", "windows_vs2022"],
    "upload": true
  },
  {
    "name": "openvino_npu_internal_tools",
    "components": ["npu_internal", "licensing"],
    "platforms": ["linux_ubuntu_22_04"],
    "upload": false
  }
]
```

A driver iterates over the manifest and, for each entry whose `platforms`
matches the current build, runs the CPack command from
[Option A](#option-a--one-archive-from-a-selected-component-set-recommended)
with `CPACK_COMPONENTS_ALL=<components>` and
`CPACK_ARCHIVE_FILE_NAME=<name>`, then publishes it when `upload` is `true`.

This manifest uses the same idea as any per-platform component list, scoped to
archive composition, and stays simple to maintain.

### Optional CMake helper `ov_cpack_add_archive`

For definitions that should live in the build system (so the archive set is
versioned with the source), a thin CMake helper can declare custom archives.
This is optional; the manifest above works without it.

```cmake
# Declares an additional archive built from an explicit CPack component set.
#   NAME       - archive file name
#   COMPONENTS - CPack components to include
#   PLATFORMS  - OS/arch tokens where the archive is produced
#   UPLOAD     - if present, the archive is published with the main archive
function(ov_cpack_add_archive)
    cmake_parse_arguments(ARG "UPLOAD" "NAME" "COMPONENTS;PLATFORMS" ${ARGN})
    # append the definition to a global property consumed at cpack time / by CI
    set_property(GLOBAL APPEND PROPERTY OV_CUSTOM_ARCHIVES
                 "${ARG_NAME}|${ARG_COMPONENTS}|${ARG_PLATFORMS}|${ARG_UPLOAD}")
endfunction()
```

Usage:

```cmake
ov_cpack_add_archive(
    NAME       openvino_runtime_min
    COMPONENTS core core_c tbb licensing
    PLATFORMS  linux_ubuntu_22_04 windows_vs2022
    UPLOAD)
```

A post-build step (or CI) reads the `OV_CUSTOM_ARCHIVES` property and runs the
matching CPack invocations. From a technical standpoint this always works with
**CPack installation components**, never with the CMake `ALL` build target; the
"default archive" stays defined as the default CPack component set for the
platform.

## Reference: public component names

Defined in `ov_define_component_names()` in
[`cmake/developer_package/packaging/packaging.cmake`](../../cmake/developer_package/packaging/packaging.cmake):

| Variable | Component | Notes |
|---|---|---|
| `OV_CPACK_COMP_CORE` | `core` | Runtime libraries and plugins |
| `OV_CPACK_COMP_CORE_C` | `core_c` | C API runtime |
| `OV_CPACK_COMP_CORE_DEV` | `core_dev` | C++ headers, cmake config |
| `OV_CPACK_COMP_CORE_C_DEV` | `core_c_dev` | C headers |
| `OV_CPACK_COMP_LICENSING` | `licensing` | License files |
| `OV_CPACK_COMP_CPP_SAMPLES` | `cpp_samples` | C++ samples |
| `OV_CPACK_COMP_C_SAMPLES` | `c_samples` | C samples |
| `OV_CPACK_COMP_PYTHON_SAMPLES` | `python_samples` | Python samples |
| `OV_CPACK_COMP_PYTHON_OPENVINO` | `pyopenvino` | Python bindings |
| `OV_CPACK_COMP_BENCHMARK_APP` | `benchmark_app` | benchmark_app tool |
| `OV_CPACK_COMP_OVC` | `ovc` | Model conversion tool |
| `OV_CPACK_COMP_PYTHON_WHEELS` | `python_wheels` | `EXCLUDE_FROM_ALL` for archives |
| `OV_CPACK_COMP_NPM` | `ov_node_addon` | `EXCLUDE_FROM_ALL` for archives |
| `OV_CPACK_COMP_INSTALL_DEPENDENCIES` | `install_dependencies` | Setup scripts |
| `OV_CPACK_COMP_SETUPVARS` | `setupvars` | `setupvars` script |
| `OV_CPACK_COMP_PKG_CONFIG` | `core_dev_pkgconfig` | pkg-config files |
| `OV_CPACK_COMP_LINKS` | `core_dev_links` | Dev symlinks |
| `npu_internal` | `npu_internal` | Not in default `ALL`; on-demand tools |

> To list the exact components available in a given build, inspect the
> generated `<build_dir>/CPackConfig.cmake` (`CPACK_COMPONENTS_ALL`) or run
> `cpack --config <build_dir>/CPackConfig.cmake --help`.
