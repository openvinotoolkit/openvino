---
name: ov-lto-issues
description: Review-only guidance for problems that appear only in LTO builds in OpenVINO, both compiler false positives and real defects (fatal with -Werror), for example the GCC -Wstringop-overflow / -Wstringop-overread triggered by inserting temporary std::string into std::unordered_set/std::unordered_map. Use when reviewing a diff or PR that matches a known pattern in this skill, when asked whether a change can trigger an LTO-only warning or error, or when a change adds or extends a warning suppression. Does not build or run anything. Production code must be fixed in code; suppression is allowed only for test targets and only for false positives. Do not use for warnings that also appear without LTO.
---

# LTO-only issues

This skill is for review only. Do not configure, build, or run tests. Judge changes from the diff and the known patterns below, and use CI results (a GCC build with `ENABLE_LTO=ON` and warnings as errors) as the build evidence. If no such CI job ran, report the build as not verified.

The same patterns are mirrored in `.github/instructions/lto-issues.instructions.md`, which the GitHub review reads. Update both when adding an entry.

## Rules

- Production code must be fixed in code. Flag any added `-Wno-*` suppression, global compiler flag, or PUBLIC flag on a shared library for production code. Use a suppression only when there is no other way, and say why.
- A suppression is acceptable only for a false positive in a test target (see [Suppress in a test target](#suppress-in-a-test-target)). Never suppress a real defect.
- Do not touch third-party sources or the `thirdparty/gtest` submodule.
- Do not assume a warning is limited to the compiler version where it was first seen. Known entries record only the versions actually verified; other versions may be affected, and older third-party suppressions in the repository show similar `stringop` / `array-bounds` false positives on other GCC versions. Check the compiler and version used by CI instead of trusting a list.

## How LTO diagnostics differ

- The warning is emitted by `lto1` during the link step, so compile-time `-Wno-*` flags and pragmas do not affect it. Only link-time flags do.
- Identical template instantiations are merged, so the message cannot name the calling function. Find the cause by reading the change or by bisecting, not from the diagnostic.
- Global flags such as `-fno-delete-null-pointer-checks` (`cmake/developer_package/compile_flags/sdl.cmake`) change what the optimizer can assume and can enable or hide these warnings. A flag added to many targets can cause different false positives elsewhere (for example `-Werror=free-nonheap-object`).
- A warning on a link line without `-Werror` is non-fatal but should still be traced to a change.

## Review a change

1. Search the diff for the patterns in [Known issues](#known-issues). Ignore patterns that do not match exactly (for example `std::set`, lvalue inserts).
2. For a CI-reported LTO warning, take the compiler, version, target, `-Werror` status, and diagnostic text from the CI log. Check that the warning does not appear in the non-LTO job.
3. Check that production code is fixed in code. Report a new suppression in production code as a finding, unless the author explains why no code fix exists.
4. Check that a test-target suppression is per target, gated by the compiler version, and has a one-line comment naming the source of the false positive.
5. Check that the replacement code keeps the behavior of the original, and that a removed suppression is backed by a CI build on the compilers it covered.
6. If the pattern is new, suggest adding it to [Known issues](#known-issues).

## Known issues

Add new entries in the same format. Keep each entry short. `Kind` is `false positive` or `real defect`; use IDs `LTO-FP-<n>` and `LTO-REAL-<n>`.

### LTO-FP-1: rvalue `std::string` inserted into a string-keyed unordered container

- **Kind:** false positive
- **Diagnostic:** `-Wstringop-overflow` / `-Wstringop-overread` from `lto1`, pointing into libstdc++ `hashtable.h` (`_M_emplace_uniq<STR>(STR&&)`, `insert(value_type&&)`, `emplace`, range `insert`) and the `std::string` move constructor.
- **Verified on:** GCC 15.2, `ENABLE_LTO=ON`, with the global `-fno-delete-null-pointer-checks`. Other GCC versions: unverified.
- **Patterns to find in a diff:**
  - `set.insert(std::move(s))`, `set.emplace(std::move(s))`, `map.emplace(std::move(key), ...)`.
  - Insert or emplace of a temporary or function result: `set.insert(std::string(...))`, `set.insert(std::regex_replace(...))`, `set.emplace(a + b)`.
  - `std::inserter(set, set.end())` or range insert with `std::make_move_iterator` over strings.
  - Ignore `std::set` / `std::map`, non-string keys, and lvalue inserts.
- **Test sources:** the `thirdparty/gtest` submodule has no note about this and must not be edited. The trigger is `suite_name_to_info_index_.emplace(std::move(test_suite_name), ...)` in `thirdparty/gtest/gtest/googletest/include/gtest/internal/gtest-param-util.h` (`ParameterizedTestSuiteRegistry`, an `unordered_map<std::string, size_t>`), reached from `INSTANTIATE_TEST_SUITE_P` / `TEST_P`. The `std::move(param_name)` insert in the same file goes into a `std::set` and is not the pattern. Do not flag every `TEST_P`; accept a per-target suppression only when CI reports the warning for that test target (seen for `ov_npu_func_tests`).
- **Production fix, in order of preference:**
  1. Construct in place from a view: `set.emplace(std::string_view{...})`.
  2. Insert a named lvalue: build into a reusable `std::string name`, then `set.insert(name); name.clear();`.
  3. Restructure so the string is not moved into the container.
  - Replacing a static container with an array does not help.
  - If Coverity raises `copy_instead_of_move` for a deliberate lvalue copy, annotate with `// coverity[ copy_instead_of_move:SUPPRESS]` only when the variable is not reused afterwards.
- **Reference fix:** `deserialize_tensor_names` in `src/core/xml_util/src/xml_deserialize_util.cpp`.

### Template for a new entry

```markdown
### LTO-<FP|REAL>-<n>: <short pattern name>

- **Kind:** <false positive | real defect>
- **Diagnostic:** <warning or error and the library/inlined frames seen>
- **Verified on:** <compiler and version, LTO setting, relevant global flags>
- **Patterns to find in a diff:** <code shapes a reviewer can grep for>
- **Production fix:** <ordered code-level fixes>
- **Reference fix:** <path of an example fix>
```

## Suppress in a test target

Add a block to that test target only, with one short comment stating where the false positive comes from:

```cmake
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    # GCC 15 false positive in <where it comes from>
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-stringop-overflow -Wno-stringop-overread)
    target_link_options(${TARGET_NAME} PRIVATE -Wno-stringop-overflow -Wno-stringop-overread)
endif()
```

- Adjust the flags and the version gate to the entry. Gate by the versions actually verified, and widen the gate when another version reproduces it.
- The link option silences warnings emitted by `lto1`; the compile option covers warnings emitted at compile time. Keep whichever the target needs, or both for uniformity.
- Examples: `src/plugins/intel_npu/tests/unit/CMakeLists.txt`, `src/plugins/intel_npu/tests/functional/CMakeLists.txt`.
- For the gtest trigger, name `gtest-param-util.h` in the comment so the source is findable.

## Evidence

- Do not build or run tests. Read the CI checks of the PR instead: a GCC job with `ENABLE_LTO=ON` and warnings as errors, covering the changed targets.
- Report the result as passing, failing, pending, or not run. A passing job covers only the compilers and targets it built.
- Without such a job, state that the fix or removed suppression is not verified and name the build the author should run (`ENABLE_LTO=ON`, `CMAKE_COMPILE_WARNING_AS_ERROR=ON`, affected targets, search the log for the warning flag such as `stringop`).
