# Overview of Inference Engine Python* API

This API provides a simplified interface for Inference Engine functionality that allows you to:

* Handle the models
* Load and configure Inference Engine plugins based on device names
* Perform inference in synchronous and asynchronous modes with arbitrary number of infer requests (the number of infer requests may be limited by target device capabilities)

## Supported OSes

Inference Engine Python\* API is supported on Ubuntu\* 18.04 and 20.04, CentOS\* 7.3 OSes, Raspbian\* 9, Windows\* 10
and macOS\* 10.x.

Supported Python* versions:

| Operating System | Supported Python\* versions: |
|:----- | :----- |
| Ubuntu\* 18.04  | 3.7 |
| Ubuntu\* 20.04  | 3.7, 3.8 |
| Windows\* 10 | 3.7, 3.8 |
| CentOS\* 7.3 | 3.7 |
| macOS\* 10.x  | 3.7 |
| Raspbian\* 9  | 3.7 |


## Set Up the Environment

To configure the environment for the Inference Engine Python\* API, run:
 * On Ubuntu\* 18.04 or 20.04: `source <INSTALL_DIR>/setupvars.sh .`
 * On CentOS\* 7.4: `source <INSTALL_DIR>/setupvars.sh .`
 * On macOS\* 10.x: `source <INSTALL_DIR>/setupvars.sh .`
 * On Raspbian\* 9,: `source <INSTALL_DIR>/setupvars.sh .`
 * On Windows\* 10: `call <INSTALL_DIR>\setupvars.bat`

The script automatically detects latest installed Python\* version and configures required environment if the version is supported.
If you want to use certain version of Python\*, set the environment variable `PYTHONPATH=<INSTALL_DIR>/python/<desired_python_version>`
after running the environment configuration script.

## API Reference
For the complete API Reference, see  [Inference Engine Python* API Reference](ie_python_api/api.html)
