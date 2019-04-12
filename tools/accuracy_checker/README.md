# Deep Learning accuracy validation framework

## Installation

### Prerequisites

Install prerequisites first:

#### 1. Python

**accuracy checker** uses **Python 3**. Install it first:

- [Python3][python3], [setuptools][setuptools]:

```bash
sudo apt-get install python3 python3-dev python3-setuptools python3-pip
```

Python setuptools and python package manager (pip) install packages into system directory by default. Installation of accuracy checker tested only via [virtual environment][virtualenv].

In order to use virtual environment you should install it first:

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

#### 2. Frameworks

The next step is installing backend frameworks for Accuracy Checker.

In order to evaluate some models required frameworks have to be installed. Accuracy-Checker supports these frameworks:

- [OpenVINO][openvino-get-started].
- [Caffe][caffe-get-started].

You can use any of them or several at a time.

#### 3. Requirements installation
```bash
pip3 install -r requirements.txt

[python3]: https://www.python.org/downloads/
[setuptools]: https://pypi.python.org/pypi/setuptools
[caffe-get-started]: accuracy_checker/launcher/caffe_installation_readme.md
[virtual-environment]: https://docs.python.org/3/tutorial/venv.html
[virtualenv]: https://virtualenv.pypa.io/en/stable
[openvino-get-started]: https://software.intel.com/en-us/openvino-toolkit/documentation/get-started