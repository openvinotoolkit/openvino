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

The `CPU functional tests` job in the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow uses this cache for sharing tests execution time to speed up the subsequent runs. First, the artefacts are saved with `actions/cache/save`:
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
with a particular key: `${{ runner.os }}-${{ runner.arch }}-tests-functional-cpu-stamp-${{ github.sha }}`. 

Then it could be seen in the [repository's cache](https://github.com/openvinotoolkit/openvino/actions/caches):
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

This cache could be used to store dependencies and large assets (models, datasets, etc.) that are to be used by different workflow jobs. 

**Note**: This cache is enabled for the Linux [self-hosted runners](./runners.md) only.

The drive is available on the self-hosted machines, and to make it available inside [the Docker containers](./docker_images.md), 
the mounting point should be added under the `container`'s `volumes` key in a job configuration:
```yaml
Build:
  ...
  runs-on: aks-linux-16-cores-32gb
  container:
    image: openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04
    volumes:
      - /mount:/mount
    options: -e SCCACHE_AZURE_BLOB_CONTAINER -e SCCACHE_AZURE_CONNECTION_STRING
```

The first `/mount` in `- /mount:/mount` is the path on the runner, the second `/mount` is the path in the Docker container by which the resources will be available.

### Available Resources

* `pip` cache
  * Accessible via an environment variable `PIP_CACHE_PATH: /mount/caches/pip/linux` defined on a workflow level
  * Used in the jobs that have Python usage 
* onnx models for tests
  * Accessible by the path: `/mount/onnxtestdata`
  * Used in the `ONNX Models tests` job in the [`linux.yml`](./../../../../.github/workflows/linux.yml) workflow
* Linux RISC-V with Conan build artefacts
  * Used in the [`linux_riscv.yml`](./../../../../.github/workflows/linux_riscv.yml) workflow

To add new resources, contact someone from the CI team for assistance.

## Cloud Storage via Azure Blob Storage

This cache is used for sharing OpenVINO build artefacts between runs. 
The [`sccache`](https://github.com/mozilla/sccache) tool can cache, upload and download build files to/from [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs).

**Note**: This cache is enabled for [self-hosted runners](./runners.md) only.

`sccache` needs several things to work:
* [be installed](#sccache-installation)
* [credential environment variables: `SCCACHE_AZURE_BLOB_CONTAINER`, `SCCACHE_AZURE_CONNECTION_STRING`](#passing-credential-environment-variables)
  * They are already set up on the self-hosted runners
  * They could be passed to a Docker containers via the `options` key under the `container` key
* [`SCCACHE_AZURE_KEY_PREFIX` environment variable](#providing-sccache-prefix) - essentially the folder where the cache for this OS/architecture will be saved
* [`CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_C_COMPILER_LAUNCHER` environment variables](#enabling-sccache-for-cc-files) to enable `sccache` for caching C++/C build files

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

The `SCCACHE_AZURE_BLOB_CONTAINER` and `SCCACHE_AZURE_CONNECTION_STRING` variables should be set in the environment so that `sccache` could communicate with Azure Blob Storage.

These variables are already set in the environments of the self-hosted runners so if a job does not use a Docker container, there is no action required for these variables.

If a job needs a [Docker container](./docker_images.md), these variables should be passed via the `options` key under the `container` key:
```yaml
Build:
  ...
  runs-on: aks-linux-16-cores-32gb
  container:
    image: openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04
    volumes:
      - /mount:/mount
    options: -e SCCACHE_AZURE_BLOB_CONTAINER -e SCCACHE_AZURE_CONNECTION_STRING
```

This way they would be available inside the container for `sccache` to use.

### Providing `sccache` Prefix

The folder on the remote storage where the cache for the OS/architecture will be saved is provided via the `SCCACHE_AZURE_KEY_PREFIX` environment variable under the job's `env` key:
```yaml
Build:
  ...
  env:
    ...
    CMAKE_CXX_COMPILER_LAUNCHER: sccache
    CMAKE_C_COMPILER_LAUNCHER: sccache
    ...
    SCCACHE_AZURE_KEY_PREFIX: ubuntu20_x86_64_Release
```

### Enabling `sccache` for C++/C Files

To tell CMake to use the caching tool, the `CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_C_COMPILER_LAUNCHER` environment variables should be set under the job's `env` key:
```yaml
Build:
  ...
  env:
    ...
    CMAKE_CXX_COMPILER_LAUNCHER: sccache
    CMAKE_C_COMPILER_LAUNCHER: sccache
    ...
    SCCACHE_AZURE_KEY_PREFIX: ubuntu20_x86_64_Release
```
or in the CMake configuration command.
