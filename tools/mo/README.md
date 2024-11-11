## Installation

### Installing from PyPi
1. Create a virtual environment and activate it, e.g.:
```
virtualenv -p /usr/bin/python3.7 .env3
source .env3/bin/activate
```

2. Install openvino-dev package, it contains model conversion API:   
```
pip install openvino-dev
```

This will download all requirements and will install MO in your current virtual environment. 
If you need only particular frameworks you can specify them manually as optional dependencies in square brackets.
E.g. the command below will install dependencies to support ONNX\* and TensorFlow2\* models:
```
pip install openvino-dev[onnx,tensorflow2]
```
To enable support of all frameworks:
```
pip install openvino-dev[all]
```
By default, if no frameworks are specified, dependencies to support ONNX\* and TensorFlow2\* are installed.

[//]: <> (### Installing wheel package from provided OpenVINO™ offline distribution)
[//]: <> (To be done)

## Converting models
* [Converting Model](../../docs/MO_DG/prepare_model/convert_model/Converting_Model.md)

## Setup development environment
If you want to contribute to model conversion API you will need to deploy developer environment. 
You can do that by following the steps below:

1. Create virtual environment and activate it, e.g.:
```
virtualenv -p /usr/bin/python3.7 .env3
source .env3/bin/activate
```

2. Clone the OpenVINO™ repository and change dir to model-optimizer
```
git clone https://github.com/openvinotoolkit/openvino
cd openvino/tools/mo/
```

3. Install openvino-mo package for development:
```
pip install -e .
```
or run `setup.py develop`, result will be the same:
```
python setup.py develop
```

This will download all requirements and deploy model conversion API for development in your virtual environment: 
specifically will create *.egg-link into the current directory in your site-packages.
As previously noted, you can also manually specify to support only selected frameworks :
```
pip install -e ".[onnx,tensorflow2]"
```

### How to run unit-tests

1. Run tests with:
<pre>
    python -m unittest discover -p "*_test.py" [-s PATH_TO_DIR]
</pre>

### How to capture unit-tests coverage

1. Run tests with:
<pre>
    coverage run -m unittest discover -p "*_test.py" [-s PATH_TO_DIR]
</pre>

2. Build html report:
<pre>
    coverage html
</pre>

### How to run code linting

1. Run the following command:
<pre>
    pylint openvino/tools/mo/ mo.py
</pre>
