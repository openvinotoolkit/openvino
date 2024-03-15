# End-to-end Tests User Documentation
This folder contains a code to run end-to-end validation of OpenVINO on real models of different frameworks (PyTorch, TensorFlow, and ONNX)

The documentation provides necessary information about environment setup for e2e validation run, adding new model to the validation, and instructions to launch validation.

> The following steps assume that your current working directory is:
> `tests/e2e_tests`

### Environment preparation:
   * Install Python modules required for tests:
       ```bash
       pip3 install -r requirements.txt 
       ```
     
### Add model from TensorFlow Hub repo to end-to-end validation:
To add new test for model from TF Hub repo just add new line into pipelines/production/tf_hub/precommit.yml
This line should contain comma separated model name and its link
```
movenet/singlepose/lightning,https://www.kaggle.com/models/google/movenet/frameworks/tensorFlow2/variations/singlepose-lightning/versions/4
```

### Main entry-point 

There is one main testing entry-point which is responsible for test run - test_base.py. This script performs the
following actions:
1. Loads model from its source 
2. Infers original model through framework
3. Converts original model through OVC convert model
4. Infers converted model through OpenVINO
5. Provides results of element-wise comparison of framework and OpenVINO inference

#### Launch tests

[test_base.py](https://github.com/openvinotoolkit/openvino/tree/master/tests/e2e_tests/test_base.py) is the main script to run end-to-end tests.
Run all end-to-end tests in `pipelines/`:
```bash
pytest test_base.py
```
`test_base.py` options:  

- `--modules=MODULES [MODULES ...]` - Paths to tests.  
- `-k TESTNAME [TESTNAME ...]`- Test names.
- `-s` - Step-by-step logging.
    
Example:
```bash
pytest test_base.py -s --modules=pipelines/production/tf_hub
```

> For full information on pytest options, run `pytest --help` or see the [documentation](https://docs.pytest.org)
