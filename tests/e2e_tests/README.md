# End-to-end Tests User Documentation

This folder contains a code to run end-to-end validation of OpenVINO on real models of different frameworks (PyTorch, TensorFlow, and ONNX). Validation steps include conversion of a real model to `ov::Model`, model inference on data samples using OpenVINO, and comparison inference results with a reference results from original framework.

The documentation provides necessary information about environment setup for e2e validation run, adding new model to the validation, and instructions to launch validation. 

## Basic scripts

There is one basic script which is responsible for test run with default pipeline - test_base.py. This script perform the
following actions:
1. Loads model from its source 
2. Collects reference from framework
3. Converts original model through OVC convert model
4. Performs inference on converted model
5. Compares reference and output from converted model through element-wise comparison


> test_base.py script is run with the help of [pytest](https://docs.pytest.org).
> The command-line is the same for all the scripts, though some of them
> __might not__ use specific options.

### Command Line

> The following steps assume that your current working directory is:
> `tests/e2e_tests`

1. Run tests:
    * Environment preparation:
        * Install Python modules required for tests:
        ```bash
        pip3 install -r requirements.txt 
        ```

   * Main entry-point  

       Module [test_base.py](https://github.com/openvinotoolkit/openvino/tree/master/tests/e2e_tests/test_base.py) is main entry-point to run E2E OSS tests.  
       Run all E2E OSS tests in `pipelines/`:
       ```
       pytest test_base.py
       ```
       `test_base.py` options:  
    
       - `--modules=MODULES [MODULES ...]` - Paths to tests.  
       - `-k TESTNAME [TESTNAME ...]`- Test names.  
       - `--env_conf=ENV_CONF` - Path to environment config.  
       - `--base_test_conf=TEST_CONF` - Path to test config.  
       - `-s` - Step-by-step logging.
    
       Run `pytest test_base.py --help` to check 'custom' command line options defined by the script.  
    
       Example:
       ```
       pytest test_base.py -s --modules=pipelines/production/tf_hub
       ```  
    >    See also Pytest Usage and Invocations https://docs.pytest.org/en/documentation-restructure/how-to/usage.html

There are several useful command-line parameters provided both by pytest and by
the scripts. For full information on pytest options, run `pytest --help` or see
the [documentation](https://docs.pytest.org). For full information on "custom"
options see the corresponding script's documentation.


# Add model from TF Hub repo:
This is the instruction how to make a new E2E for TF Hub model, follow the next instructions to add new E2E from ticket
1. To add new test for model from TF Hub repo just add new line into tests/e2e_tests/pipelines/production/tf_hub/nightly.yml
This line should contain at least two params: model name, and it's link to download
