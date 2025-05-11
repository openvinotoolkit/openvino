# Caches

OpenVINO uses caches to accelerate builds and tests while minimizing network usage.

## Table of Contents

* [Available Caches](#available-caches)
* [GitHub Actions Cache](#github-actions-cache)
* [Shared Drive Cache](#shared-drive-cache)
* [Cloud Storage via Azure Blob Storage](#cloud-storage-via-azure-blob-storage)


## Available Caches

Three types of caches are available:
* [GitHub Actions cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
  * Available for both GitHub-hosted and self-hosted runners.
  * Accessible by `actions/cache` action.
  * Limited to 10 GB per repository.
  * Suitable for small dependencies caches and reusable artifacts.
* [Shared drive cache](#shared-drive-cache-usage-and-structure)
  * Available only to self-hosted runners.
  * Automatically available via a predefined path.
  * Large storage.
  * Suitable for large caches, such as build caches, models, and datasets.
* Cloud storage via [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs)
  * Available only to self-hosted runners.
  * Used to cache and share build artifacts with [`sccache`](https://github.com/mozilla/sccache).

Jobs in the workflows utilize these caches based on their requirements.

## GitHub Actions Cache

This cache is used for sharing small dependencies or artifacts between runs.
Refer to the [GitHub Actions official documentation](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
for a complete reference.

The `CPU functional tests` job in the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml)
workflow uses this cache for sharing test execution time to speed up the subsequent runs.
First, the artifacts are saved with `actions/cache/save` with a particular
key `${{ runner.os }}-${{ runner.arch }}-tests-functional-cpu-stamp-${{ github.sha }}`:
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

Then it appears in the [repository's cache](https://github.com/openvinotoolkit/openvino/actions/caches):
![gha_cache_example](../../assets/CI_gha_cache_example.png)

The following runs can download the artifact from the repository's cache with `actions/cache/restore`
and use it:
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
The `restore-keys` key is used to find the required cache entry. `actions/cache` searches for
a full or partial match and downloads the located cache to the provided `path`.

Refer to the [actions/cache documentation](https://github.com/actions/cache) for a complete syntax reference.

## Shared Drive Cache

This cache is used to store dependencies and large assets, such as models and datasets,
that will be used by different workflow jobs.

>**NOTE**: This cache is enabled for Linux [self-hosted runners](./runners.md) only.

The drive is available on self-hosted machines. To make it available inside [Docker containers](./docker_images.md),
add the mounting point under the `container`'s `volumes` key in a job configuration:
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

In `- /mount:/mount`, the first `/mount` is the path on the runner, the second `/mount` is the
path in the Docker container where the resources will be available.

### Available Resources

* `pip` cache
  * Accessible via the environment variable `PIP_CACHE_PATH: /mount/caches/pip/linux`, defined at the workflow level
  * Used in jobs that involve Python usage
* onnx models for tests
  * Accessible at the path: `/mount/onnxtestdata`
  * Used in the `ONNX Models tests` job in the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml) workflow
* Linux RISC-V with Conan build artifacts
  * Used in the [`linux_riscv.yml`](./../../../../.github/workflows/linux_riscv.yml) workflow

To add new resources, contact a member of the CI team for assistance.

## Cloud Storage via Azure Blob Storage

This cache is used for sharing OpenVINO build artifacts between runs.
The [`sccache`](https://github.com/mozilla/sccache) tool can cache, upload and download build files to/from [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs).

>**NOTE**: This cache is enabled for [self-hosted runners](./runners.md) only.

`sccache` requires several configurations to work:
* Installation. Refer to the [sccache installation](#sccache-installation) section.
* [Credential environment variables: `SCCACHE_AZURE_BLOB_CONTAINER`, `SCCACHE_AZURE_CONNECTION_STRING`](#passing-credential-environment-variables)
  * The variables are already set up on the self-hosted runners.
  * The variables can be passed to a Docker container via the `options` key under the `container` key.
* [`SCCACHE_AZURE_KEY_PREFIX` environment variable](#providing-sccache-prefix) to specify the folder where the cache for the current OS/architecture will be saved.
* [`CMAKE_CXX_COMPILER_LAUNCHER` and `CMAKE_C_COMPILER_LAUNCHER` environment variables](#enabling-sccache-for-cc-files) to enable `sccache` for caching C++/C build files

### `sccache` Installation

The installation is done via the community-provided `mozilla-actions/sccache-action` action:
```yaml
- name: Install sccache
  uses: mozilla-actions/sccache-action@v0.0.3
  with:
    version: "v0.5.4"
```

This step must be placed in the workflow **before** the build step.

### Passing Credential Environment Variables

The `SCCACHE_AZURE_BLOB_CONTAINER` and `SCCACHE_AZURE_CONNECTION_STRING` variables must be
set in the environment to enable `sccache` communication with Azure Blob Storage.

These variables are already set in the environment for jobs on self-hosted runners
without a Docker container, requiring no further actions.


If a job needs a [Docker container](./docker_images.md), pass the variables via the `options`
key under the `container` key to make them accessible for `sccache` inside the container:
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

### Providing `sccache` Prefix

The folder on the remote storage where the cache for the OS/architecture will be saved is
provided via the `SCCACHE_AZURE_KEY_PREFIX` environment variable under the job's `env` key:
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

To instruct CMake to use the caching tool, set the `CMAKE_CXX_COMPILER_LAUNCHER`
and `CMAKE_C_COMPILER_LAUNCHER` environment variables under the job's `env` key:
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
You can also set the options in the CMake configuration command.
