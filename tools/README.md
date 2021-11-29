# OpenVINO™ Python* openvino.tools package

## General
`openvino.tools` package includes:
* openvino.tools.benchmark

Please, refer to https://docs.openvinotoolkit.org for details.

## Installation
Choose necessary Python\* version and define `PYTHONPATH` environment variable.

### Prerequisites

Install prerequisites first:

#### 1. Python

**openvino.tools** is **Python 3** library. Install it first:

- [Python3][python3]
- [setuptools][setuptools]

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

#### 2. Install package prerequisites

The next step is installing package prerequisites.

```bash
python3 -m pip install -r benchmark/requirements.txt
```

### Configuration

Each subpackage has specific configuration. Please, refer to specific subpackage documentation for details.

[python3]: https://www.python.org/downloads/
[setuptools]: https://pypi.python.org/pypi/setuptools

