# Overview of the Caches used in the OpenVINO GitHub Actions CI

To speed up builds and tests and reduce network usage, OpenVINO workflows use caches.

## Available Caches

Three types of caches are available:
* [GitHub Actions cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
  * Available for both GitHub-hosted and self-hosted runners
  * Accessible by `actions/cache` action
  * Limited to 10 GB per repository
  * Suitable for small dependencies caches and artefacts that could be reused between runs 
* [Shared drive cache](#shared-drive-cache-usage-and-structure)
  * Available only to the self-hosted runners
  * Automatically available via a certain path
  * Large storage
  * Suitable for large caches
    * e.g., build caches, models, datasets
* Cloud storage via [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs)
  * Available only to the self-hosted runners
  * Used to cache and share build artefacts with [`sccache`](https://github.com/mozilla/sccache)

The jobs in the workflows utilize appropriate caches based on job's needs.

## GitHub Actions Cache

This cache is used for sharing small dependencies or artefacts between runs. Refer to the [GitHub Actions official documentation](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows) for a complete reference.

The `CPU functional tests` job in the `linux.yml` workflow uses this cache for sharing tests execution time to speed up the subsequent runs. First, the artefacts are saved with `actions/cache/save`:
```yaml
CPU_Functional_Tests:
  name: CPU functional tests
  ...
  steps:
    - name: Save tests execution time
      uses: actions/cache/save@v3
      if: github.ref_name == 'master'
      with:
        path: ${{ env.PARALLEL_TEST_CACHE }}
        key: ${{ runner.os }}-${{ runner.arch }}-tests-functional-cpu-stamp-${{ github.sha }}
    ...
```
with a particular key: `${{ runner.os }}-${{ runner.arch }}-tests-functional-cpu-stamp-${{ github.sha }}`. Then it could be seen in the [repository's cache](https://github.com/openvinotoolkit/openvino/actions/caches):
![gha_cache_example](../../../sphinx_setup/_static/images/ci/gha_cache_example.png)

The next runs could download the artefact from the repository's cache with `actions/cache/restore` and use it:
```yaml
CPU_Functional_Tests:
  name: CPU functional tests
  ...
  steps:
    - name: Restore tests execution time
      uses: actions/cache/restore@v3
      with:
        path: ${{ env.PARALLEL_TEST_CACHE }}
        key: ${{ runner.os }}-${{ runner.arch }}-tests-functional-cpu-stamp-${{ github.sha }}
        restore-keys: |
          ${{ runner.os }}-${{ runner.arch }}-tests-functional-cpu-stamp
    ...
```
The `restore-keys` key is used to find the needed cache entry. `actions/cache` searches for the full or partial match and downloads the found cache to the provided `path`.

Refer to the [`actions/cache`'s documentation](https://github.com/actions/cache) for a complete syntax reference.

## Shared Drive Cache Usage and Structure

**Note**: This cache is enable for [self-hosted runners](./runners.md) only.

## Cloud Storage via Azure Blob Storage

This cache is used for sharing OpenVINO build artefacts between runs. 
The [`sccache`](https://github.com/mozilla/sccache) tool can cache, upload and download build files to/from [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs).

**Note**: This cache is enable for [self-hosted runners](./runners.md) only.

`sccache` needs several things to work:
* be installed
* credential environment variables: `SCCACHE_AZURE_BLOB_CONTAINER`, `SCCACHE_AZURE_CONNECTION_STRING`
  * They are already set up on the self-hosted runners
  * They could be passed to a Docker containers via the `options` key under the `container` key. See the example below
* `SCCACHE_AZURE_KEY_PREFIX` environment variable - essentially the folder where the cache for this OS/architecture will be saved
* `CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_C_COMPILER_LAUNCHER` environment variables to enable `sccache` for caching C++/C build files

### `sccache` Installation

The installation is done via the community-provided `mozilla-actions/sccache-action` action:
```yaml
- name: Install sccache
  uses: mozilla-actions/sccache-action@v0.0.3
  with:
    version: "v0.5.4"
```

This step should be placed somewhere in the workflow **before** the build step.

### Passing Credential Environment Variables

If a job is running outside a container, the 

### Providing `sccache` Prefix

### Enabling `sccache` for 
