# Package BOM Tests

## Purpose

A package Bill of Materials (BOM) records the installed file paths of a CPack
component. BOM tests protect the production package's public file structure and
layout: accidentally adding, removing, or renaming a packaged file can break
users even if compilation and functional tests pass.

The reviewed reference manifests (GOLD BOMs) are version-controlled under
`boms/<configuration>/<component>.json`. Each CI configuration has its own
baseline because operating systems, architectures, and build variants can
legitimately produce different layouts.

A developer making an intentional package-layout change must update the affected
GOLD BOMs in the same PR. This update is their explicit approval of the added,
removed, or renamed package files. Do not update a GOLD BOM just to make an
unexpected packaging regression pass.

BOM tests compare file paths, not file contents, hashes, or dependency licenses.
JSON formatting and file-list ordering do not affect the comparison.

## CMake Generation

Generation is implemented in:

* [`packaging.cmake`](../../../../cmake/developer_package/packaging/packaging.cmake):
  CPack component registration through `ov_cpack_add_component()` and
  collection through `ov_cpack()`.
* [`bom.cmake`](../../../../cmake/developer_package/packaging/bom.cmake):
  component accumulation and creation of the aggregate `bom` and per-component
  `ov_bom_<component>` build targets.
* [`bom_generate.cmake`](../../../../cmake/developer_package/packaging/bom_generate.cmake):
  isolated component installation and generation of sorted relative file paths.

`ENABLE_BOM_GENERATION` is **ON by default**. The normal build generates
`<build-directory>/bom/<platform>/<component>.json` for the registered CPack
components. The platform directory is normally `linux`, `windows`, or `macos`.
Each manifest has this structure:

```json
{
  "component": "core",
  "files": [
    "runtime/include/openvino/openvino.hpp"
  ]
}
```

To disable generation, configure with `-DENABLE_BOM_GENERATION=OFF`. To generate
manifests explicitly, run `cmake --build <build-directory> --target bom`, or use
the `ov_bom_<component>` target for one component.

Generation supports archive `CPACK_GENERATOR` values: `7Z`, `TBZ2`, `TGZ`,
`TXZ`, `TZ`, `TZST`, and `ZIP`. It is automatically disabled when a non-archive
generator is selected (for example, `DEB`, `RPM`, or `NPM`).

The comparison described below runs **only in GitHub Actions**; it is not part
of the default CMake build or CTest.

## GitHub Actions Checks

After a successful build, the build workflow publishes a
`bom_<configuration>` artifact. A separate BOM test job invokes
[`job_bom_tests.yml`](../../../../.github/workflows/job_bom_tests.yml), which
downloads that exact artifact and runs
[`compare.py`](../../../../.github/scripts/bom/compare.py).

For example, `bom_ubuntu_22_04_x86_64` contains `linux/core.json`, while its
reference is `boms/ubuntu_22_04_x86_64/core.json`. The platform directory inside
the artifact is not repeated in the GOLD directory.

The checks run in this order:

1. Verify that the required BOM artifact exists and contains a platform
   directory with component JSON manifests. Missing, empty, or malformed
   artifacts fail the test.
2. Find `boms/<configuration>` in the checked-out revision. If the entire
   configuration is missing, fail the test and request baseline initialization.
   Generated manifests must still be valid.
3. Compare component names. All new and removed components are reported as
   failures, with instructions to add or remove the corresponding GOLD JSON
   files. File-list comparison is skipped for this configuration until the
   component lists agree.
4. If component names agree, compare the file paths in every component.
   Report all added and removed paths and fail each affected component's test.
   Invalid manifests also fail the test.

The comparison collects failures rather than stopping at the first changed
component. Diagnostics include a link to this page. The
`test-results-bom_<configuration>` artifact contains `TEST-bom.xml` (JUnit) and
`TEST-bom.html`, uploaded even when the comparison fails. Download the artifact
and open the HTML report in a browser to see every check, its status, and the
full list of added and removed paths. Missing GOLD configurations are failures,
not skipped checks, in both reports and make the job fail.

File-list differences identify the component and the direction relative to GOLD,
for example:

```text
Component 'cpu' differs from its GOLD BOM:
  deleted runtime/lib/old-library.so on the PR branch
  added runtime/lib/new-library.so on the PR branch
```

These details appear in the job log, annotations, JUnit XML, and HTML report.
A rename is reported as a deleted path and an added path: path-only BOMs cannot
reliably distinguish a rename from two independent changes.

### Build Artifacts and Reruns

The Build job uploads `openvino_package` and `bom_<configuration>` before the
separate BOM test job starts. A failed BOM comparison does not remove or
overwrite these artifacts. Reports use a separate `test-results-bom_<configuration>`
artifact.

After **Re-run failed jobs**, GitHub's Summary may display artifacts only from
the selected attempt. If Build was not rerun, select the original build attempt
to download its package and generated BOM. No duplicate package upload or custom
platform Summary is needed.

### Pull Request Notifications

When a platform detects changed components or file paths,
[`bom_pr_comment.yml`](../../../../.github/workflows/bom_pr_comment.yml) posts
a comment for that platform with a link to the BOM job and this documentation.
Reruns update the same comment rather than adding duplicates. A subsequent
successful BOM check updates it to show that the differences have been resolved.
Setup failures and missing GOLD configurations remain job errors, but are not
reported as detected file-layout changes.

This notifier runs on `workflow_run` using trusted default-branch code and
GitHub API metadata, without executing PR code or downloading PR artifacts.
It supports fork PRs and becomes active after the notifier workflow is merged
into the default branch.

Coverage and Linux Level Zero builds are excluded from BOM tests because they
are special-purpose, non-product builds whose packages are not distributed
externally. BOM tests protect the layout of publicly distributed product
packages, not these internal build variants.

Coverage, Linux Level Zero, Ubuntu 22.04 DPC++, Ubuntu 24.04 SYCL, and Windows Level Zero builds
pass `generate-bom: false` to the reusable build
workflow, which sets `ENABLE_BOM_GENERATION=OFF` and skips BOM artifact upload.
The `generate-bom` input defaults to `true` for other Linux and Windows builds.

## Updating an Existing GOLD BOM

1. Open the failed BOM test job and review every reported component and file
   change. Correct any unintended packaging changes first.
2. Download the matching `bom_<configuration>` artifact from the workflow
   run's **Summary > Artifacts**.
3. For an intentional change, copy the affected `<component>.json` manifests
   from the artifact's platform directory into `boms/<configuration>`.
   Add manifests for new components and remove manifests for removed
   components. A rename is an addition and a removal.
4. Commit these GOLD changes in the originating PR. Inspect their diff as the
   explicit approval of the changed public package layout.
5. Repeat for every affected CI configuration and rerun the checks.

The test never overwrites GOLD manifests automatically.

## Initializing a New Configuration

A new platform, architecture, or build configuration can be added on any
development or release branch. Its first build has no GOLD baseline, so the BOM
job fails until the configuration's baseline is added through a reviewed PR.

Use the separately provided
[Initial GOLD BOM baseline workflow](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/ci/github_actions/overview.md#initial-gold-bom-baseline)
to collect the build artifacts and create the initial baseline PR. Once those
manifests are merged into the relevant branch, subsequent builds compare
against them normally.
