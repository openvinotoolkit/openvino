# OpenVINO Provider

This is a custom GitHub Action that provides pre-built OpenVINO artifacts for a given revision — either a specific commit,
the latest available from a branch, or a publicly available package.
It is designed for use in GitHub Action workflows of third-party repositories that depend on OpenVINO and require 
validation against it, avoiding the need to rebuild OpenVINO from source.

## Usage
A simple example: calling OpenVINO provider in a separate job within a workflow:
```yaml
  openvino_download:
    name: Download prebuilt OpenVINO
    outputs:
      status: ${{ steps.openvino_download.outcome }}
      ov_wheel_source: ${{ steps.openvino_download.outputs.ov_wheel_source }}
      ov_version: ${{ steps.openvino_download.outputs.ov_version }}
      docker_tag: ${{ steps.get_docker_tag.outputs.docker_tag }}
    timeout-minutes: 10
    defaults:
      run:
        shell: bash
    runs-on: aks-linux-medium
    container:
      image: 'openvinogithubactions.azurecr.io/openvino_provider:0.1.0'
      volumes:
        - /mount:/mount
        - ${{ github.workspace }}:${{ github.workspace }}

    steps:
    - uses: openvinotoolkit/openvino/.github/actions/openvino_provider@master
      id: openvino_download
      with:
        platform: 'ubuntu22'
        revision: latest_available_commit
```
### Main input arguments
* `revision` - specifies which OpenVINO version to download artifacts for. Accepted values:
  * `latest_available_commit` - returns the latest complete _post-commit_ artifacts built via OpenVINO GHA workflows.
  * `<specific commit hash>` / `HEAD` - returns _post-commit_ artifacts for a specific commit hash or HEAD, 
  if available.
  * `latest_nightly` - fetches the latest available nightly artifacts from 
  [storage.openvinotoolkit.org](https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly)
  * `<specific package version>` (e.g. `2025.1` or `2024.4.0rc2`) - for any other release/pre-release/custom package 
  from [storage.openvinotoolkit.org](https://storage.openvinotoolkit.org/repositories/openvino/packages)


**Note**: OpenVINO provider _execution_ is supported on Linux GitHub runners only; though it can provide build artifacts for Windows as well - they can be propagated to Windows runners via GitHub storage (see [example](https://github.com/openvinotoolkit/openvino_tokenizers/blob/77f7abc0a900b2f189f397576a1f03aa9ffab383/.github/workflows/windows.yml#L98) in openvino_tokenizers). To download _post-commit_ artifacts, 
the job with the action must run on an [Azure-hosted Linux runner](./runners.md), e.g. `runs-on: aks-linux-medium`
in the example above.


* `platform` - specifies the operating system for which artifacts should be downloaded. 
Available options are listed in the input description
[here](../../../../.github/actions/openvino_provider/action.yml).
* `arch` - specifies the target architecture (defaults to x86_64). Valid values are listed 
in the input description [here](../../../../.github/actions/openvino_provider/action.yml).
* `install_dir` - if specified, downloads artifacts to a given local path; otherwise artifacts are getting installed 
to a root of GitHub workspace and then uploaded to GitHub artifacts storage 
(uploading to GitHub storage is useful when reusing artifacts in multiple different jobs or propagating 
them to incompatible runners such as Windows or GitHub-hosted ones, where this action cannot run).

### Main outputs
Information about downloaded artifacts is provided via action's outputs:

* `ov_artifact_name` - the name under which artifacts are uploaded to GitHub storage
(only applicable when a local install_dir is not set).
* `ov_wheel_source` - a string to pass as an option to `pip install` command to specify the location of 
OpenVINO Python wheels (either --find-links or --extra-index-url).
* `ov_version` - the OpenVINO version associated with the downloaded artifacts 
(can be used to explicitly specify the version during installation; or for reporting purposes).

### Example outputs usage:
```yaml
  openvino_tokenizers_tests:
    name: OpenVINO tokenizers tests
    needs: [ openvino_download ]
    defaults:
      run:
        shell: bash
    runs-on: ubuntu-22.04
      steps:
      ...
      - name: Download OpenVINO package
        uses: actions/download-artifact@95815c38cf2ff2164869cbab79da8d1f422bc89e # v4.2.1
        with:
          name: ${{ needs.openvino_download.outputs.ov_artifact_name }}
          path: ${{ env.INSTALL_DIR }}
          merge-multiple: true

      - name: Install OpenVINO Python wheel from pre-built artifacts
        run: |
          python3 -m pip install openvino==${{ needs.openvino_download.outputs.ov_version }} ${{ needs.openvino_download.outputs.ov_wheel_source }}
        working-directory: ${{ env.INSTALL_DIR }}
```

For details on available inputs and outputs, refer to the action definition:
[.github/actions/openvino_provider/action.yml](../../../../.github/actions/openvino_provider/action.yml).

See also a real-life example in the openvino_tokenizers repo:
* [Artifacts download job](https://github.com/openvinotoolkit/openvino_tokenizers/blob/77f7abc0a900b2f189f397576a1f03aa9ffab383/.github/workflows/linux.yml#L31-L55)
* [Artifacts usage (Python)](https://github.com/openvinotoolkit/openvino_tokenizers/blob/77f7abc0a900b2f189f397576a1f03aa9ffab383/.github/workflows/linux.yml#L265-L276)
