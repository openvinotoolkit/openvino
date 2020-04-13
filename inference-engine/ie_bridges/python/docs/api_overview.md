# Overview of Inference Engine Python* API

This API provides a simplified interface for Inference Engine functionality that allows you to:

* Handle the models
* Load and configure Inference Engine plugins based on device names
* Perform inference in synchronous and asynchronous modes with arbitrary number of infer requests (the number of infer requests may be limited by target device capabilities)

## Supported OSes

Inference Engine Python\* API is supported on Ubuntu\* 16.04 and 18.04, CentOS\* 7.3 OSes, Raspbian\* 9, Windows\* 10 
and macOS\* 10.x.    
Supported Python* versions:  

| Operating System | Supported Python\* versions: |
|:----- | :----- |
| Ubuntu\* 16.04  |  2.7, 3.5, 3.6, 3.7 |
| Ubuntu\* 18.04  |  2.7, 3.5, 3.6, 3.7 |
| Windows\* 10 |  3.5, 3.6, 3.7 |
| CentOS\* 7.3 | 3.4, 3.5, 3.6, 3.7  |
| macOS\* 10.x  | 3.5, 3.6, 3.7 |   
| Raspbian\* 9  | 3.5, 3.6, 3.7 |   


## Set Up the Environment

To configure the environment for the Inference Engine Python\* API, run:
 * On Ubuntu\* 16.04 or 18.04 CentOS\* 7.4: `source <INSTALL_DIR>/bin/setupvars.sh .`
 * On CentOS\* 7.4: `source <INSTALL_DIR>/bin/setupvars.sh .`
 * On macOS\* 10.x: `source <INSTALL_DIR>/bin/setupvars.sh .`
 * On Raspbian\* 9,: `source <INSTALL_DIR>/bin/setupvars.sh .`
 * On Windows\* 10: `call <INSTALL_DIR>\bin\setupvars.bat`

The script automatically detects latest installed Python\* version and configures required environment if the version is supported.  
If you want to use certain version of Python\*, set the environment variable `PYTHONPATH=<INSTALL_DIR>/python/<desired_python_version>`
after running the environment configuration script.

## API Reference
For the complete API Reference, see  [Inference Engine Python* API Reference](ie_python_api.html)
