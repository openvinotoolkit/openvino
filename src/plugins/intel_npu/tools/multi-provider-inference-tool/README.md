# multi-provider-inference-tool

Yet another one implementation of single-image-test, which is conceived with intention to launch nn-mod inference using using different frameworks

# Setup virtual environment

##### 1. STANDALONE MODE

If you intend to use the tool in standalone mode and expect that all necessary dependencies will be installed from Python default repositories, please change dir to the tool directory and execute the following commands:

    setup_venv_standalone.cmd <Python venv name>
    
You may omit `<Python venv name>`, in this case a default venv will be created

Having this done, please proceed to the [Usage](#usage) section

##### 2. OpenVINO package MODE

In case if you have already got OpenVINO+TOOLS packages, then you might probably want this tool to use applications from these packages. In this case, please execute the following commands from the tool directory:

##### Windows
    
    setup_venv_standalone.cmd <OpenVINO package path> <Python venv name>

Where: 

`<OpenVINO package path>` is the OpenVINO package where `setupvars.bat` is specified

You may omit `<Python venv name>`, in this case a default venv will be created


Having this done, please proceed to the [Usage](#usage) section

# Usage

Please run ```mpit.py --help``` if you are not familiar with the tool.

1. To list all supported providers just specify a model by its path using the CMD argument `-m ` :

```
    mpit.py -m <model path>
```

 and look at a help message. The list of supported providers/backend will be enlisted in an extended description of the important parameter  ```-p, --provider PROVIDER```, as it depicted as an example below:

```
      -p, --provider PROVIDER
                        Inference provider, available: ['onnx/CPUExecutionProvider', 'onnx/OVEP/CPU', 'onnx/OVEP/NPU', 'ov/CPU',
                        'ov/GPU', 'ov/NPU']
```
2. To show a model description specify a valid provider by the CMD argument `-p`:

```
mpit.py -m <model path> -p <provider>
```
Information about I/O of model in JSON format will be printed there
Additionally, this information will also be stored as a JSON file by path
`<provider>/<model_name>.json`. You could use this file to extract a model path and its input-output descriptions, which will be helpful in order to constitute additional CMD arguments later

3. To launch simple inference, specify one or more source files per model input. Please pay attention that a name of a particular model input is one of major parameters in source files descriptions.
There are two options how to specify source files understandably by this tool

3.1.  Sources as images:

```
mpit.py -m <model_path> -p <provider> -i
{\"<input_name>\": {\"files\": [\"<file_path_0>\", \"<file_path_1>\"], \"type\": \"image\"}}
```

 3.2. Sources as binary files

```
mpit.py -m <model_path> -p <provider> -i
 {\"<input_name>\": {\"files\": [\"<file_path_0>\", \"<file_path_1>\"], \"type\": \"bin\", \"shape\": [1, 3, 299, 299], \"layout\": \"NCHW\", \"element_type\": \"float32\"}}
```

As you can see using the binary format as a source file requires more additional arguments to be specified.
In order to mitigate this inconvenience, the tool stores last consumed inputs in JSON files which can be located at paths:
- for images
```<provider>\0\<model_name>\inputs_img.json```
- for binary blobs
```<provider>\0\<model_name>\inputs_dump_data.json```

So that you could re-use these files as source files arguments in a next tool invocation:

```mpit.py -m <model_path> -p <provider> -i <input_img.json>```

or
```mpit.py -m <model_path> -p <provider> -i <input_dump_data.json>```

respectively.


4. It is possible to specify a model pre-postprocessing by using the additional CMD parameter `-ppm`, which must receive the same JSON format as it specified in a model info JSON

# Testing

#### Test locally

Please install these requirements required by tests:

```pip install -r tests/requirements.txt```

Then execute 

```python -m unittest```

#### Test in a docker image [TODO]
    `docker build -f tests/Dockerfile --build-arg https_proxy=<your https proxy> --build-arg http_proxy=<your http proxy> --build-arg no_proxy=<your no proxy> -t sit_tests:latest .`

