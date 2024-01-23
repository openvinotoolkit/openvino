# Overview of the Custom GitHub Actions used in the OpenVINO GitHub Actions CI

Several actions are written specifically for the needs of the OpenVINO workflows.

You can find all the custom actions and their source code [here](../../../../.github/actions).

Refer to the official [custom GitHub Action documentation](https://docs.github.com/en/actions/creating-actions/about-custom-actions) for more information.

## Available Custom Actions

* [Setup Python](#setup-python)
* [System Info Print](#system-info-print)
* Smart CI (see details: [feature documentation](./smart_ci.md))

## Setup Python

This custom action installs the required Python version and environment variables on the runner. 

Under the hood it uses the GitHub-provided `actions/setup-python` and community-provided `deadsnakes-action` depending on the machine architecture. 
`actions/setup-python` does not work on the Linux ARM64 machines so `deadsnakes-action` is used instead.

### Usage
```yaml
  - name: Setup Python ${{ env.PYTHON_VERSION }}
    uses: ./openvino/.github/actions/setup_python
    with:
      version: '3.11'
      pip-cache-path: ${{ env.PIP_CACHE_PATH }}
      should-setup-pip-paths: 'true'
      self-hosted-runner: 'true'
      show-cache-info: 'true'
```
where:
* `version` - the Python version to install in the `MAJOR.MINOR` format
* `pip-cache-path` - the path to the `pip` cache on the mounted share. Read more about shares and caches [here](./caches.md)
* `should-setup-pip-paths` - whether the action should set up the `PIP_CACHE_DIR` and `PIP_INSTALL_PATH` environment variables for later usage
* `self-hosted-runner` - whether the runner is self-hosted. Read more about available runners [here](./runners.md)
* `show-cache-info` - whether the action should show the share space occupied by the `pip` cache

## System Info Print

This custom action prints the system information in the standard output:
* Operating system
* Machine architecture
* CPU, RAM and memory information

Works on Linux, macOS and Windows.

### Usage
```yaml
  - name: System info
    uses: ./openvino/.github/actions/system_info
```
