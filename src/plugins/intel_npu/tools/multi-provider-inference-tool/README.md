# multi-provider-inference-tool

This toolset enables launching neural network model inference using different frameworks (referred to as **providers**, e.g., OpenVINO and ONNX) to produce inference artifacts like blobs and metadata. These artifacts can be used for metrics gathering, allowing users to evaluate the capabilities of various AI software stacks and hardware accelerators.
Providers can be loaded dynamically as plugins, so users are not required to install all provider dependencies. The application remains operational as long as at least one inference provider is set up (e.g., OpenVINO).

# Setup virtual environments

There are two approaches for setting up the tool: **"standalone" mode** and **"OpenVINO package" mode**.

### 1. STANDALONE MODE

If you want to use the tool in standalone mode and expect that all necessary dependencies will be installed from Python's default repositories, navigate to the tool directory and execute the following command:

#### Windows

    setup_venv_standalone.cmd <Python venv name>

You may omit this `<Python venv name>`, in which case a default venv will be created.

On first execution, the script sets up a Python virtual environment and installs all required dependencies for all supported providers. The initial setup may take some time, as it downloads required packages from PyPI.
Once this is complete, proceed to the [Activate a virtual environment](#activate-a-virtual-environment) section.

### 2. OpenVINO package MODE

If you already have the OpenVINO+Tools packages, you can configure the tool to use the existing provider binaries and existing Python wheels from those packages. In this case, run the following command in the tool directory:

#### Windows

    setup_venv_ov_package.cmd <OpenVINO package path> <Python venv name>

Where:

`<OpenVINO package path>` is the path to the OpenVINO package containing `setupvars.bat`.

You may also omit this `<Python venv name>` to allow the script to create a default venv.
On first execution, the script configures a Python virtual environment and installs the tool’s dependencies as well as Python wheels from the specified OpenVINO package.
Having this done, please proceed to the [Activate a virtual environment](#activate-a-virtual-environment) section

# Activate a virtual environment

Once the virtual environment is set up, activate it using:

    <Python venv name>/Scripts/activate

Upon activation, a set of unit tests will run to validate the toolset.
If all tests pass, proceed to the [Usage](#usage) section.

> Note: An additional Python virtual environment named .venv_tests will
> be created automatically to handle test dependencies. If you wish to
> run the tests manually, activate this environment. You do not need to
> activate the environments created in the previous steps for this.

# Usage

Please run ```mpit.py --help``` if you are unfamiliar with the tool.

1. To list all supported providers, specify a model path with the `-m` or `--model` argument:

```
    mpit.py -m <model path>
```

The help message will display the available providers/backends in the extended description of the `-p` or `--provider` argument:

```
    -p, --provider PROVIDER
                        An inference provider, available:
                                onnx/CPUExecutionProvider$
                                onnx/OpenVINOExecutionProvider/CPU$
                                onnx/OpenVINOExecutionProvider/NPU((\.(.+)$)|$)
                                ov/CPU$
                                ov/GPU((\.(.+)$)|$)
                                ov/NPU((\.(.+)$)|$)
```

As you can see, available providers are specified using regular expressions, allowing fine-grained selection of hardware accelerators.
For example:

 - To choose a second GPU for inference using OpenVINO, use ov/GPU.1.
 - To specify a particular NPU generation, append its generation number:
ov/NPU.4010 targets a device from the "LunarLake" generation.

2. To display model information such as input/output names and layout, specify a valid provider using the `-p` argument:

```
mpit.py -m <model path> -p <provider>
```


The model's I/O information will be printed in JSON format and saved as:

`<provider>/<model_name>.json`.

This file contains details needed to construct the `-i` / `--inputs` argument (e.g., input names and formats).

3. To run inference, specify one or more source files for each model input using the `-i` or `--inputs` argument.

> Note: The name of each model input is a required parameter when
> specifying source files.
> You can find input names, expected shapes, and data types in the JSON
> file from the previous step.

There are several ways to specify inference sources. Refer to the documentation for further details.

 -  3.1.  Sources as images:

```
mpit.py -m <model_path> -p <provider> -i
"{\"<input_name>\": {\"files\": [\"<file_path_0>\", \"<file_path_1>\"], \"type\": \"image\", \"convert\":{\"shape\": <a new desired shape>, "\layout"\: <a new desired layout>, \"element_type\":<a new desired data type>}}}"
```

> Note: The "convert" field is optional. If not specified, the tool will
> use the model's native "shape", "layout", and "element_type" as default values.

Refer to the [JSON schema](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu/tools/multi-provider-inference-tool/schema/input_source.json?raw=true) for full details.


 -  3.2. Sources as binary files

```
mpit.py -m <model_path> -p <provider> -i
 {\"<input_name>\": {\"files\": [\"<file_path_0>\", \"<file_path_1>\"], \"type\": \"bin\", \"shape\": [1, 3, 299, 299], \"layout\": \"NCHW\", \"element_type\": \"float32\"}}
```
Please take look at the [JSON schema](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu/tools/multi-provider-inference-tool/schema/input_source.json) for more information.

As shown, using binary files as input sources requires explicitly specifying additional fields: `shape`, `layout`, and `element_type`.
To simplify repeated usage, the tool automatically stores metadata of the last used inputs in model-local JSON files:

- for images
```<provider>\0\<model_name>\inputs_img.json```
- for binary blobs
```<provider>\0\<model_name>\inputs_dump_data.json```

You can reuse these files in future runs by passing them directly as arguments to the `-i` option:

```mpit.py -m <model_path> -p <provider> -i <input_img.json>```

or
```mpit.py -m <model_path> -p <provider> -i <input_dump_data.json>```

respectively.


4. You can specify model pre- and post-processing using the `-ppm` argument. This argument must contain a JSON object matching the structure provided in the model info JSON, and must conform to the [JSON schema](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu/tools/multi-provider-inference-tool/schema/model.json)

# Testing

If you have already followed the [ Setup virtual environments](#setup-virtual-environments) section, simply activate the test virtual environment:

```.venv_tests/Scripts/activate```

Then, run the test suite:

```python -m unittest```

This will execute the embedded Python unit tests.

If you prefer a clean environment or skipped the setup step earlier, you can use the following script to create a standalone test environment::

```tests\run_tests.cmd <Python venv name>```

- If you omit <Python venv name>, a temporary virtual environment is created. It runs the tests and is then removed automatically.

- If you provide <Python venv name>, the environment is preserved and can be reused for future test runs.
