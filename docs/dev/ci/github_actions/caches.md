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
* [Shared drive cache](#shared-drive-cache)
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
      uses: actions/cache/save@27d5ce7f107fe9357f9df03efb73ab90386fccae # v5.0.5
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
      uses: actions/cache/restore@27d5ce7f107fe9357f9df03efb73ab90386fccae # v5.0.5
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
  needs: Docker
  runs-on: <self-hosted runner>
  container:
    image: ${{ fromJSON(needs.docker.outputs.images).ov_build.<image> }}
    volumes:
      - <shared drive path>:<shared drive path>
      - <secrets path on runner>:/secrets:ro
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

In `- <shared drive path>:<shared drive path>`, the first path is the location of the shared drive
on the runner, the second one is the path in the Docker container where the resources will be
available. Use the same mount point as the existing workflows for the target runner.

### Available Resources

* `pip` cache
  * Accessible via the `PIP_CACHE_PATH` environment variable, defined at the workflow level
  * Used in jobs that involve Python usage
* onnx models for tests
  * Stored in a dedicated folder on the shared drive
  * Used in the `ONNX Models tests` job in the [`ubuntu_22.yml`](./../../../../.github/workflows/ubuntu_22.yml) workflow
* Linux RISC-V with Conan build artifacts
  * Used in the [`linux_riscv.yml`](./../../../../.github/workflows/linux_riscv.yml) workflow

To add new resources, contact a member of the CI team for assistance.

### Hugging Face Model Cache

Hugging Face model-test jobs call `setup_hf_proxy` after checking out the action
and making `python3` available:

```yaml
steps:
  - name: Setup Hugging Face proxy
    uses: ./.github/actions/setup_hf_proxy
```

The action exports `HF_ENDPOINT=http://hf-proxy.hf-proxy.svc.cluster.local`
and `HF_HUB_DISABLE_XET=1`. It sets `HF_HOME` to `${{ runner.temp }}/hf-home`
and both `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` to
`${{ runner.temp }}/hf-home/hub`, overriding shared-cache settings for subsequent
steps. Call it in each reusable workflow that needs the proxy; reusable workflows
do not inherit the caller's environment.

Runner selection and existing Hugging Face authentication settings are unchanged.
DMZ GPU runners retain the original Hugging Face endpoint and shared cache at
`/mnt/caches/huggingface`. GPU RoPE tests do not call `setup_hf_proxy`. LLM accuracy
tests use the proxy only when the caller sets `use-hf-proxy: true`, independently
of `use-pip-proxy`. The AKS CPU job opts in; the DMZ GPU jobs use the default
of `false`.

Runners using `setup_hf_proxy` must be able to resolve and reach this cluster-local
endpoint. There is no
direct-download fallback if the proxy is unreachable. Models occupy local runner
disk even when served from the proxy cache. The proxy covers Hugging Face Hub
clients, not other model hosts such as Kaggle or torchvision.

### Remote Storage Compiler Cache

Builds can share compiled objects through ccache's remote storage on the shared drive
by setting `CCACHE_SECONDARY_STORAGE` (or `CCACHE_REMOTE_STORAGE` on newer ccache) to a
`file://` URL. Each build configuration uses its own directory inside the shared storage.
Users:
* The RISC-V GNU and Conan builds in [`linux_riscv.yml`](./../../../../.github/workflows/linux_riscv.yml)
  and [`linux_riscv_conan.yml`](./../../../../.github/workflows/linux_riscv_conan.yml).
* The Windows builds in
  [`job_build_windows.yml`](./../../../../.github/workflows/job_build_windows.yml) and
  [`windows_conditional_compilation.yml`](./../../../../.github/workflows/windows_conditional_compilation.yml).

ccache cannot cache MSVC compilations with a separate PDB (`/Zi`), so the Windows Debug build
sets `-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded` (`/Z7`), which stores debug information
in object files. Builds with embedded debug information link without `/INCREMENTAL`, which keeps
PDB files free of stale incremental-link data. See `ENABLE_STATIC_LIBS_PDB` in
[CMake options for custom compilation](./../../cmake_options_for_custom_compilation.md) for how
static libraries are handled.

`CCACHE_MAXSIZE` limits only the local cache. The remote storage is trimmed periodically by the
`Cleanup_remote_ccache_*` jobs in
[`cleanup_caches.yml`](./../../../../.github/workflows/cleanup_caches.yml) with
`ccache --trim-dir --trim-method mtime` (ccache 4.8 or newer), which evicts the least recently used
entries down to `CCACHE_REMOTE_MAX_SIZE`. Builds must keep `update-mtime=true` in the remote storage
URL so that cache hits refresh the modification time used for eviction.

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
  needs: Docker
  runs-on: <self-hosted runner>
  container:
    image: ${{ fromJSON(needs.docker.outputs.images).ov_build.<image> }}
    volumes:
      - <shared drive path>:<shared drive path>
      - <secrets path on runner>:/secrets:ro
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
    SCCACHE_AZURE_KEY_PREFIX: ${{ inputs.os }}_${{ inputs.arch }}_Release
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
    SCCACHE_AZURE_KEY_PREFIX: ${{ inputs.os }}_${{ inputs.arch }}_Release
```
You can also set the options in the CMake configuration command.
