---
applyTo: "**/*.{cpp,hpp,h,cc,cxx},**/CMakeLists.txt,**/*.cmake"
---

# LTO-only issues (review checklist)

Some problems appear only in LTO builds (`ENABLE_LTO=ON`), and CI does not build with LTO, so catch suspected code in review. Report each as a suspected issue on the exact line; it is confirmed only by an LTO build. Full details and the registry of known issues: `.agents/skills/ov-lto-issues/SKILL.md`. Keep this checklist and that registry in sync.

Never suggest suppressing a real defect or any production code; suppression is only for false positives in test targets, scoped per target (`PRIVATE`) with a one-line comment.

## LTO-FP-1: rvalue `std::string` into a string-keyed unordered container (false positive)

GCC 15 with `ENABLE_LTO=ON` can fail the build with a false-positive `-Wstringop-overflow` / `-Wstringop-overread`.

Flag in added or changed C++ lines an rvalue `std::string` inserted into a string-keyed `std::unordered_set` / `std::unordered_map`:

- `set.insert(std::move(s))`, `set.emplace(std::move(s))`, `map.emplace(std::move(key), ...)`
- insert or emplace of a temporary or function result: `set.insert(std::string(...))`, `set.insert(std::regex_replace(...))`, `set.emplace(a + b)`
- `std::inserter(set, set.end())` or range insert through `std::make_move_iterator` over strings

Do not flag ordered `std::set` / `std::map`, non-string keys, inserts of a named lvalue, or `emplace(std::string_view{...})`.

Flag in CMake a new `-Wno-stringop-*` (or similar warning suppression) or `-fdelete-null-pointer-checks` on a production target, a global scope, or a PUBLIC/INTERFACE scope.

Fix production code with `set.emplace(std::string_view{...})`, or build into a reusable `std::string name` and insert the lvalue (`set.insert(name); name.clear();`).

## Real LTO defects

None recorded yet. Add each as `LTO-REAL-<n>` with its patterns to flag and the code fix.
