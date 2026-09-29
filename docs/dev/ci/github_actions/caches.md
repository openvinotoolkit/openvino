# Caches

OpenVINO uses caches to accelerate builds and tests while minimizing network usage.

## Table of Contents

* [Available Caches](#available-caches)
* [GitHub Actions Cache](#github-actions-cache)
* [Shared Drive Cache](#shared-drive-cache)
* [`ccache` Remote Storage](#ccache-remote-storage)
* [Cloud Storage via Azure Blob Storage](#cloud-storage-via-azure-blob-storage)


## Available Caches

Three types of caches are available:
* [GitHub Actions cache](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
  * Available for both GitHub-hosted and self-hosted runners.
  * Accessible by `actions/cache` action.
  * Limited to 10 GB per repository.
  * Suitable for small dependencies caches and reusable artifacts.
* [Shared drive cache](#shared-drive-cache)
  * Available only to self-hosted runners.
  * Automatically available via a predefined path.
  * Large storage.
  * Suitable for large caches, such as build caches, models, and datasets.
* Cloud storage via [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs)
  * Available only to self-hosted runners.
  * Used to cache and share build artifacts with [`sccache`](https://github.com/mozilla/sccache).
* [`ccache` remote storage](#ccache-remote-storage) on the shared drive
  * Available only to self-hosted runners.
  * Used to cache and share build artifacts with [`ccache`](https://ccache.dev) on Windows and Ubuntu 24.04.

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
      - /home/runner/secrets/:/secrets:ro
    options: -e SCCACHE_AZURE_BLOB_CONTAINER
  steps:
    - name: Append the environment variable - load SCCACHE_AZURE_CONNECTION_STRING from file
      shell: bash
      run: |
        SCCACHE_AZURE_CONNECTION_STRING="$(cat /secrets/sccache/connection-string)"
        echo "::add-mask::${SCCACHE_AZURE_CONNECTION_STRING}"
        echo "SCCACHE_AZURE_CONNECTION_STRING=${SCCACHE_AZURE_CONNECTION_STRING}" >> $GITHUB_ENV
        echo "✓ Connection string loaded and masked"
  ...
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

## `ccache` Remote Storage

The Windows pipelines ([`job_build_windows.yml`](./../../../../.github/workflows/job_build_windows.yml),
[`windows_conditional_compilation.yml`](./../../../../.github/workflows/windows_conditional_compilation.yml))
and the Ubuntu 24.04 pipeline ([`ubuntu_24.yml`](./../../../../.github/workflows/ubuntu_24.yml))
cache C++/C build files with [`ccache`](https://ccache.dev) using its
[remote storage](https://ccache.dev/manual/latest.html#_remote_storage_backends) `file` backend
pointed at the shared drive. Every compilation queries the job-local cache first and the shared
directory second, so no cache archive has to be restored before or uploaded after the build.

The configuration is done entirely via environment variables under the job's `env` key:
```yaml
Build:
  ...
  env:
    CMAKE_CXX_COMPILER_LAUNCHER: ccache
    CMAKE_C_COMPILER_LAUNCHER: ccache
    CCACHE_REMOTE_STORAGE: "file:///mount/caches/ccache_remote/ubuntu_24_04_x86_64_Release|umask=002|update-mtime=true"
    CCACHE_DIR: ${{ github.workspace }}/ccache
    CCACHE_TEMPDIR: ${{ github.workspace }}/ccache_temp
    CCACHE_MAXSIZE: 3G
    CCACHE_BASEDIR: ${{ github.workspace }}
    CCACHE_SLOPPINESS: pch_defines,time_macros
```
On Windows, the shared drive is mounted at `C:\mount`, so the URL takes the form
`file:///C:/mount/caches/ccache_remote/<prefix>`.

Notes:
* Cache entries are content-addressed and the directory is neither keyed by commit nor by branch,
  so every pull request, every commit within a pull request and every post-commit run read from
  and write to the same cache. Do not add the branch name or `github.sha` to the path.
* `CCACHE_BASEDIR` makes `ccache` hash absolute paths below the workspace as relative ones, and
  `CCACHE_SLOPPINESS` keeps `__DATE__`/`__TIME__` and precompiled headers out of the hash. Without
  them the same source compiles to a different cache entry on another runner or on another day.
* `ccache` never evicts entries from its remote storage. The
  [`cleanup_caches.yml`](./../../../../.github/workflows/cleanup_caches.yml) workflow removes
  entries that have not been used for 30 days; `update-mtime=true` is what makes that eviction
  least-recently-used.
* `umask=002` keeps the entries writable for every user of the shared drive.

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
      - /home/runner/secrets/:/secrets:ro
    options: -e SCCACHE_AZURE_BLOB_CONTAINER
  steps:
    - name: Append the environment variable - load SCCACHE_AZURE_CONNECTION_STRING from file
      shell: bash
      run: |
        SCCACHE_AZURE_CONNECTION_STRING="$(cat /secrets/sccache/connection-string)"
        echo "::add-mask::${SCCACHE_AZURE_CONNECTION_STRING}"
        echo "SCCACHE_AZURE_CONNECTION_STRING=${SCCACHE_AZURE_CONNECTION_STRING}" >> $GITHUB_ENV
        echo "✓ Connection string loaded and masked"
  ...
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
