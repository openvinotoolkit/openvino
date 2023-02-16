# OpenVINO™ Python* development tools

## General
OpenVINO includes following tools:
* openvino.tools.benchmark

Please, refer to https://docs.openvino.ai for details.

## Installation

### Prerequisites

Install prerequisites first:

#### 1. Python

Install **Python** prerequisites:

- [Python3][python3]
- [setuptools][setuptools]

Run following command to install these prerequisites on Ubuntu*:
```bash
sudo apt-get install python3 python3-dev python3-setuptools python3-pip
```

Python setuptools and python package manager (pip) install packages into system directory by default. There are several options:

- work inside [virtual environment][virtualenv] (best solution).
- use `--user` option for all `pip` commands.
- install all dependencies with *sudo* permissions.

In order to use virtual environment you should install it:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv -p `which python3` <directory_for_environment>
```

Before starting to work inside virtual environment, it should be activated:

```bash
source <directory_for_environment>/bin/activate
```

Virtual environment can be deactivated using command

```bash
deactivate
```

#### 2. Install packages

You can install tools by specifying path to tool with `setup.py` in `pip install` command:

```bash
python3 -m pip install <tools_folder>/
```
For example, to install Benchmark Tool, use the following command:  
```bash
python3 -m pip install benchmark_tool/
  ```

### Configuration

Each subpackage has specific configuration. Please, refer to specific subpackage documentation for details.

[python3]: https://www.python.org/downloads/
[setuptools]: https://pypi.python.org/pypi/setuptools

