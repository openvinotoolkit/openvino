# E2E OSS tests

This folder contains E2E test framework code, E2E tests and related materials
(reference outputs, test data, etc.)

## Basic scripts

There are two basic scripts currently available to the user. They perform the
following actions:

1. Run E2E OSS tests
2. Run E2E OSS reference collection (references are usually stored locally!)

> These scripts are run with the help of [pytest](https://docs.pytest.org).
> The command-line is the same for all the scripts, though some of them
> __might not__ use specific options.

### Command Line

> The following steps assume that your current working directory is:
> `tests/e2e_oss`

1. Run tests:
    * Environment preparation:
        * Install Python modules required for tests:
        ```bash
        pip3 install -r requirements.txt 
        ```

    * Running:
    ~~~bash
    pytest test.py
    ~~~

2.  Run (non-runtime) reference collection:
    ~~~bash
    pytest collect_refs.py
    ~~~

There are several useful command-line parameters provided both by pytest and by
the scripts. For full information on pytest options, run `pytest --help` or see
the [documentation](https://docs.pytest.org). For full information on "custom"
options see the corresponding script's documentation.

## Contents

The following is the list of the main components of the E2E OSS tests.

    e2e_oss/
    |__ test.py                 Main entry-point to run pytest tests
    |__ env_config_local.yml    Default environment configuration file
    |__ test_config_local.yml   Default tests configuration file
    |__ base_test_rules.yml     Test rules configuration file for base e2e tests

    |__ pipelines/              Test pipelines definitions
    |__ plugins/                Local plugins for pytest

    utils/e2e/
    |__ common/                 Common framework components (i.e. BaseProvider)
    |__ comparator/             Reference/Inference Engine results comparators
                                (i.e. classification, object detection)
    |__ infer/                  Inference engine runners
    |__ ir_provider/            IR providers (i.e. OpenVINO converter)
    |__ postprocessor/          Inference *output* processors
    |__ preprocessor/           Inference *input* processors
    |__ readers/                File readers (i.e. .npy, .npz readers)
    |__ ref_collector/          Reference collectors (TensorFlow, PyTorch ...)


# Add model from TF Hub repo:
This is the instruction how to make a new E2E for TF Hub model, follow the next instructions to add new E2E from ticket
1. To add new test for model from TF Hub repo just add new line into tests/e2e_oss/pipelines/production/tf_hub/nightly.yml
This line should contain at least two params: model name and it's link to download


## Add new model

This is the instruction how to make a new E2E test with ticket (with [CVS-38924](https://jira.devtools.intel.com/browse/CVS-38924) as an example).

#### Load model
1. Go to Epic Link.
2. Find the model files in attachment.
3. Check the model description and if the model is private.
    
    1. Private model
    
        All private models should be commited to [private-models](https://gitlab-onezoo.toolbox.iotg.sclab.intel.com/onezoo/models-private).
        Make a new dir in destination folder there, upload the model and add model.yml with description.
        When MR is merged, need to run [job](https://openvino-ci.intel.com/job/administration/job/update-models/) to update the models.
        > Please request permissions to models-private repository using AGS [here](https://ags.intel.com/identityiq/accessRequest/accessRequest.jsf#/accessRequest/manageAccess/add?filterKeyword=onezoo%20gitlab) (OneZoo Gitlab Private Models)    
        
        Example: 
        
            The model was originally taken from: https://jira.devtools.intel.com/secure/attachment/1437854/tlv3_n0_2x.pb
            Ticket: https://jira.devtools.intel.com/browse/CVS-38924
            
    2. Non-private model
    
        Non-private models should be uploaded to shared disk `/nfs/inn/proj/vdp/vdp_tests/models/internal`.
        In description find the information about the model in order to go to destined folder, make new dir if it is necessary and upload the model file there. 
        > Note that you may not have write permissions there.

#### Test Model Optimizer conversion

To make sure MO can generate Intermediate Representation (further IR) from the original model you should start MO by passing the model and its input string.
* Check if in the model description contains command for [MO](https://github.com/openvinotoolkit/openvino/blob/master/tools/mo/openvino/tools/mo/mo.py) conversion and use it.  

    Example:
    ```
    ./mo.py --input_model tlv3_n0_2x.pb --input "cur_inputs[1 256 256 3],pre_inputs[1 256 256 3],pre_outputs[1 512 512 3]"
    ```
* Otherwise contact model enabling owner mentioned in JIRA Epic in order to get the command.  

  > As a last resort you can try to run mo.py yourself.
                                                                                                                                                                                                   
    *Usage mo.py*
                                                                                                                                                                                                                                                                                                                                                                                                                                           
    `--input_model INPUT_MODEL`: file with a pre-trained model or proto file with model weights.   
    `--input INPUT`: Quoted list of comma-separated input nodes names with shapes, data types, and values for freezing. The shape and value are specified as space-separated lists (e.g  "inputs[1 255 255], outputs[1]")
    
    *Useful options*  
 
    `--output_dir OUTPUT_DIR, -o OUTPUT_DIR`: Directory that stores the generated IR.  
    `--compress_to_fp16 {True,False}`: Convert data to FP16 formant for all intermediate tensors and weights. It is turned on by default.  
    `--disable_fusing`: Turn off fusing of linear operations to Convolution  
    `--log-cli-level {CRITICAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}`: logging type (--log-cli-level=DEBUG:  step-by-step logging while test running).

If MO successfully generates IR, proceed with next step.
    
#### Prepare input data

In general, input data is .npz file that is contained in model framework [input data folder](https://github.com/intel-innersource/frameworks.ai.openvino.tests/tree/master/e2e_oss/test_data/inputs). File is the dictionary where keys are names of input parameters and values are numpy arrays with input data. The specific shapes and filling of arrays depend on the model and you may find it out in model description.  
> According to [Epic](https://jira.devtools.intel.com/browse/CVS-30729) Topaz Video Super Resolution [input file](https://github.com/intel-innersource/frameworks.ai.openvino.tests/blob/master/e2e_oss/test_data/inputs/tf/video_sr.npz) (will be merged soon) has the following structure:  
  `cur_inputs.npy`:         Numpy array (256, 256, 3) containing an original randomly selected image .                                                                                                                                                                                                                                                                                                                                                                                                                                               
  `pre_inputs.npy`:         Same array as cur_inputs.npy.                                                                                                                                                                                                                                                                                                                                                                                                                                               
  `pre_outputs.npy`:        Numpy array (512, 512, 3) containing resized image.                                                                                                                                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                                                                                                                                                 
#### Add test module

[The example test file](https://github.com/intel-innersource/frameworks.ai.openvino.tests/blob/master/e2e_oss/pipelines/production/tf/light/topaz_video_sr.py).

E2E-test is a Python class containing following attribute
```python
__is_test_config__ = True
```
1. Make new dir with .py file in model framework [pipelines folder](https://github.com/intel-innersource/frameworks.ai.openvino.tests/tree/master/e2e_oss/pipelines/production).
2. Add the test class inheriting pipelines.pipeline_base_classes.common_base_class.CommonConfig with attribute `__is_test_config__ = True` to the file. Add to class following `__init__` method:
    
    ```python
    def __init__(self, batch, device, precision, **kwargs):
    ```
3. Add reference pipeline 
    * `ref_pipeline` is test class method that is responsible for preprocessing and getting the output of the reference model. The function returns `OrderedDict` with info required to process ref model.  

        Example:  
        ```python
        self.ref_pipeline = OrderedDict([
                ("read_input", {"npz": {"path": "/path/to/input/data"}}),
                ('preprocess', OrderedDict([ 
                    ("align_with_batch", {"batch": 1}), # add batch dimension to input data
                ])),
                ('get_refs', {'score_tf': {'model': "/path/to/model")}}),
                ("postprocessor", OrderedDict([
                    ("remove_layer", {"layers_to_remove": ["name_of_layer_that_should_be_removed"]}), # if ref and ie layers does not match
                    ("align_with_batch", {"batch": self.batch}), # compare with batch size
                ]))
            ])
    
        ```
      
    * You can use [special templates](https://github.com/intel-innersource/frameworks.ai.openvino.tests/tree/master/e2e_oss/pipelines/pipeline_templates) for all required pipeline steps:
        
        * Input templates
        * Comparators templates
        * Infer templates
        * IR generation templates
        * Postprocess templates
        * Preprocess templates
        * Collect reference templates
        
        Example:  
        ```python
        self.ref_pipeline = OrderedDict([
                read_npz_input(path="/path/to/input/data"),
                assemble_preproc(batch=1),
                get_refs_tf(model="/path/to/model"),
                ("postprocessor", OrderedDict([
                    ("remove_layer", {"layers_to_remove": ["name_of_layer_that_should_be_removed"]}), # if ref and ie layers does not match
                    ("align_with_batch", {"batch": self.batch}), # compare with batch size
                ]))
            ])
    
        ```

    * You can also use [pre-stored references](https://github.com/intel-innersource/frameworks.ai.openvino.tests/tree/master/e2e_oss/test_data/references) with specifing `pre_collection` attribute.

        Example for TensorFlow:  
        ```python
        self.ref_collection = {'pipeline': OrderedDict([
                read_npz_input(path="/path/to/input/data"),
                assemble_preproc_tf(batch=1, h=h, w=w, **preproc),
                get_refs_tf(model="/path/to/model", **score_tf_args)		
        ])
                'store_path': "/path/to/model/reference",
                'store_path_for_ref_save': "/path/to/model/reference/in/repo"
        }
        ```
      
        > The specific implementation depends on the model framework.
4. Add `ie_pipeline` class method  

    `ie_pipeline` is method that is responsible for preprocessing and getting the output of the generated IR. The function returns OrderedDict similar to ref_pipeline method.
    
    Example:
    ```python

    self.ie_pipeline = OrderedDict([
            read_npz_input(path="/path/to/input/data"),
            assemble_preproc(batch=batch, permute_order=(2, 0, 1), **preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision, **self.additional_mo_args),
            common_infer_step(batch=batch, device=device),
            ("postprocessor", OrderedDict([
                ("permute_shape", {"order": (0,2,3,1)}),
            ]))
        ])
    ```
5. Write comparator class method.

    `comparator` is method that is responsible for the comparison of reference and inference engine results and test result representation.
    Example:
    ```python
    self.comparators = ssim_comparators(precision=precision, device=device)
    ```
    ##### Comparators
       
    * *"classification"* compares reference and IE models results for top-N classes (usually, top-1 or top-5). Basic result example: list of 1000 class probabilities for ImageNet classification dataset.  
    * *"dummy"* no comparison performed: test is always passed.
    * *"eltwise"* takes reference and inference results with same shape and counts max absolute and relative element-wise difference.
    * *"eltwise_kaldi"* takes reference and inference results with same shape and counts mean squared absolute and relative element-wise difference.
    * *"object_detection"* takes reference and inference sets of bounding boxes with its classes and computing [IoU](https://en.wikipedia.org/wiki/Jaccard_index) for objects with equal class.
    * *"semantic_segmentation"* takes reference and inference matrices of image size that contain a class number for every image pixel and counts relative error. 
    * *"ssim"* takes reference and IE image tensors and counts mean [structure similarity](https://en.wikipedia.org/wiki/Structural_similarity) among all channels.
    * *"ssim_4d"* takes reference and IE multidimensional image tensors and counts mean [structure similarity](https://en.wikipedia.org/wiki/Structural_similarity) among all channels and dimensions. 

#### Add filters  

[base_test_rules.yml](https://github.com/intel-innersource/frameworks.ai.openvino.tests/blob/master/e2e_oss/base_test_rules.yml) - Test rules configuration file for base pipelines. Controls which tests will be run by applying specified rules to all discovered tests and filtering out non-conforming ones.
[reshape_test_rules.yml](https://github.com/intel-innersource/frameworks.ai.openvino.tests/blob/master/e2e_oss/reshape_test_rules.yml) - Test rules configuration file for reshape tests. Controls which tests will be run by applying specified rules to all discovered tests and filtering out non-conforming ones.
[dynamism_test_rules.yml](https://github.com/intel-innersource/frameworks.ai.openvino.tests/blob/master/e2e_oss/dynamism_test_rules.yml) - Test rules configuration file for dynamism tests. Controls which tests will be run by applying specified rules to all discovered tests and filtering out non-conforming ones.

Rules specification:

- `rules`  specifies rules to be applied to tests. For example, (CPU, FP32). Rule states that for CPU device, only FP32 precision is expected. Thus, any other configurations like (CPU, FP16), (CPU, INT8), etc. are to be excluded from parameters setup for testing.  
    
    Any value in rules may represent a list of values:  
    ```
    "device: [GPU, OTHER], precision: [FP32, FP16]",  
    "model: [TF_Amazon_RL_LSTM, TF_DeepSpeech]"...
    ```
 - `filter_by`  specifies which parameters are not comparable and must be handled in a special way when applying rules. For example, rules for CPU must not affect other devices (GPU, MYRIAD, ...). Specifying "filter_by: device" means: "if device !=CPU/GPU/..., do not apply CPU/GPU/... rules to it". Same logic is useful when dealing with specific models.  
     
    One can specify multiple filters the following way:  
    ```
    "filter_by: [device, precision]"
    ```
Example:

```python
rules: [
    ...
    { model: TF_TopazVideoSR, device: [CPU, GPU]}  # Only CPU and GPU were requested (CVS-38924)
]
```

#### Run test locally
* Environment variables

    Specify PYTHONPATH and LD_LIBRARY_PATH variables:
    ```
    export LD_LIBRARY_PATH=/.../openvino/bin/intel64/Release/lib
    export PYTHONPATH=/.../openvino/bin/intel64/Release/lib/python_api/python3.6/
    ```
    
* Main entry-point  

    Module [test.py](https://github.com/intel-innersource/frameworks.ai.openvino.tests/blob/master/e2e_oss/test.py) is main entry-point to run E2E OSS tests.  
    Run all E2E OSS tests in `pipelines/`:
    ```
    pytest test.py
    ```
    `test.py` options:  
    
    - `--modules=MODULES [MODULES ...]` - Paths to tests.  
    - `-k TESTNAME [TESTNAME ...]`- Test names.  
    - `--env_conf=ENV_CONF` - Path to environment config.  
    - `--base_test_conf=TEST_CONF` - Path to test config.  
    - `--tf_models_version=VERSION` - Specify TensorFlow models version.  
    - `-s` - Step-by-step logging.
    
    Run `pytest test.py --help` to check 'custom' command line options defined by the script.  
    
    Example:
    ```
    pytest test.py -s --modules=pipelines/production/tf/topaz
    ```  
    > See also Pytest Usage and Invocations https://docs.pytest.org/en/documentation-restructure/how-to/usage.html
  
  
## FAQ

* I want to add a new test, what are the basic steps?

    1. If tests will run by default (e.g. `pytest test.py`):
        * Create new tests under `pipelines/` folder.
        * Use `test_data/` folder if you need to add new test inputs, collect
          references, etc.
    2. If test you want to add is test for model from tf hub repo:
        * just add new line in tests/e2e_oss/pipelines/production/tf_hub/nightly.yml file.
        * It should contain at least two params: model name and model link
    3. Otherwise:
        * Create separate top-level folder: i.e. `e2e_oss/custom_pipelines`
        * Create new tests there (similarly to `pipelines/` tests)
        * Use command-line options of [test.py](test_base.py) to specify location of
          your tests:
            ~~~bash
            pytest test.py --modules <path to tests>
            ~~~

* How do I update default environment for my specific case?

    1. Update environment configuration file - [env_config_local.yml](env_config_local.yml):
        ~~~yml
        models: /default/models/location
        # my tests-specific models:
        my_models: /my/models/location
        ~~~

    2. Use the following in your tests:
        ~~~python
        Environment.env['models']  # for default models
        Environment.env['my_models']  # for my specific models
        ~~~

* How do I use __different__ environment for the same tests?

    1. Write your own environment configuration similar to
       [env_config_local.yml](env_config_local.yml)

    2. Specify your environment config via command-line when running tests:
        ~~~bash
        pytest test.py --env_conf <path to env_conf>
        ~~~

* I'd like to add new device, precision, ... to testing, what are the options?

    1. If the change must be a default option, update
       [test_config_local.yml](test_config_local.yml)

    2. If the change is temporary or relevant only to some subset of tests,
       configs, write a separate `test_config.yml` and specify it explicitly
       when running tests:
        ~~~bash
        pytest test.py --test_conf <path to test_conf>
        ~~~

* I need to add new functionality to test framework, what do I do?

    1. Decide what part your functionality belongs to: comparators,
       preprocessing, postprocessing, ...

    2. Update existing files or create new ones

    3. Example of preprocessor that subtracts mean values from input data:
        ~~~python
        class SubtractMeanValues(ClassProvider):
            __action_name__ = "subtract_mean_values"  # keyword that is used
            # in tests to apply this preprocessor to data

            def __init__(self, config):
                self.mean_values = config["mean_values"]

            def apply(self, data):
                for layer in data.keys():
                    data[layer] = data[layer] - self.mean_values
                return data
        ~~~

    4. The code above will be executed if added into the pipeline:
        ~~~python
        class Test:
            __is_test_config__ = True

            def __init__(self, batch, device, precision):
                self.pipeline = ("preprocess", OrderedDict([(
                    # keyword "subtract_mean_values" that is equal to
                    # __action_name__ in SubtractMeanValues class
                    "subtract_mean_values", {"mean_values": (104, 117, 123)}
                )]))
        ~~~
