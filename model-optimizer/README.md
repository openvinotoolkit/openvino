## Project structure

Project structure:
<pre>
    |-- root
        |-- docs
            |-- Model_Optimizer_Developer_Guide - md files of documentation for the Model Optimizer
                |-- img
                |-- prepare_model
                    |-- convert_model
                    |-- customize_model_optimizer
        |-- extensions
            |-- front/caffe
                |-- CustomLayersMapping.xml.example - example of file for registering custom Caffe layers in 2017R3 public
                manner
        |-- mo
            |-- back - Back-End logic: contains IR emitting logic
            |-- front - Front-End logic: contains matching between Framework-specific layers and IR specific, calculation
            of output shapes for each registered layer
            |-- graph - Graph utilities to work with internal IR representation
            |-- middle - Graph transformations - optimizations of the model
            |-- pipeline - Sequence of steps required to create IR for each framework
            |-- utils - Utility functions
        |-- tf_call_ie_layer - Sources for TensorFlow fallback in Inference Engine during model inference
        |-- mo.py - Centralized entry point that can be used for any supported framework
        |-- mo_caffe.py - Entry point particularly for Caffe
        |-- mo_mxnet.py - Entry point particularly for MXNet
        |-- mo_tf.py - Entry point particularly for TensorFlow
        |-- ModelOptimizer - Entry point particularly for Caffe that contains same CLI as 2017R3 publicly released
        Model Optimizer
</pre>

## Prerequisites

Model Optimizer requires:

1. Python 3 or newer

2. [Optional] Please read about use cases that require Caffe available on the machine (:doc:`caffe_dependency`).
   Please follow the steps described (:doc:`caffe_build`).

## Installation instructions

1. Go to the Model Optimizer folder:
<pre>
    cd PATH_TO_INSTALL_DIR/deployment_tools/model_optimizer/model_optimizer_tensorflow
</pre>

2. Create virtual environment and activate it. This option is strongly recommended as it creates a Python sandbox and
   dependencies for Model Optimizer do not influence global Python configuration, installed libraries etc. At the same
   time, special flag ensures that system-wide Python libraries are also available in this sandbox. Skip this
   step only if you do want to install all Model Optimizer dependencies globally:

    * Create environment:
        <pre>
          virtualenv -p /usr/bin/python3.6 .env3 --system-site-packages
        </pre>
    * Activate it:
      <pre>
        . .env3/bin/activate
      </pre>
3. Install dependencies. If you want to convert models only from particular framework, you should use one of
   available <code>requirements_*.txt</code> files corresponding to the framework of choice. For example, for Caffe use
   <code>requirements_caffe.txt</code> and so on. When you decide to switch later to other frameworks, please install dependencies
   for them using the same mechanism:
   <pre>
    pip3 install -r requirements.txt
    </pre>

4. [OPTIONAL] If you use Windows OS, most probably you get python version of `protobuf` library. It is known to be rather slow,
   and you can use a boosted version of library by building the .egg file (Python package format) yourself,
   using instructions below (section 'How to boost Caffe model loading') for the target OS and Python, or install it
   with the pre-built .egg (it is built for Python 3.4, 3.5, 3.6, 3.7):
    <pre>
        python3 -m easy_install protobuf-3.6.1-py3.6-win-amd64.egg
    </pre>

   It overrides the protobuf python package installed by the previous command.

   Set environment variable to enable boost in protobuf performance:
    <pre>
        set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
    </pre>


## Command-Line Interface (CLI)

The following short examples are framework-dependent. Please read the complete help
with --help option for details across all frameworks:
<pre>
    python3 mo.py --help
</pre>

There are several scripts that convert a model:

1. <code>mo.py</code> -- universal entry point that can convert a model from any supported framework

2. <code>mo_caffe.py</code> -- dedicated script for Caffe models conversion

3. <code>mo_mxnet.py</code> -- dedicated script for MXNet models conversion

4. <code>mo_tf.py</code> -- dedicated script for TensorFlow models conversion

<code>mo.py</code> can deduce original framework where input model was trained by an extension of
the model file. Or <code>--framework</code> option can be used for this purpose if model files
don't have standard extensions (<code>.pb</code> - for TensorFlow models, <code>.params</code> - for MXNet models,
<code>.caffemodel</code> - for Caffe models). So, the following commands are equivalent::

<pre>
    python3 mo.py --input_model /user/models/model.pb
    python3 mo.py --framework tf --input_model /user/models/model.pb
</pre>
The following examples illustrate the shortest command lines to convert a model per
framework.

### Convert TensorFlow model

To convert a frozen TensorFlow model contained in binary file <code>model-file.pb</code>, run
dedicated entry point <code>mo_tf.py</code>:

    python3 mo_tf.py --input_model model-file.pb

### Convert Caffe model

To convert a Caffe model contained in <code>model-file.prototxt</code> and <code>model-file.caffemodel</code> run
dedicated entry point <code>mo_caffe.py</code>:
<pre>
    python3 mo_caffe.py --input_model model-file.caffemodel
</pre>


### Convert MXNet model

To Convert an MXNet model in <code>model-file-symbol.json</code> and <code>model-file-0000.params</code> run
dedicated entry point <code>mo_mxnet.py</code>:
<pre>
    python3 mo_mxnet.py --input_model model-file
</pre>

> **NOTE**: for TensorFlow* all Placeholder ops are represented as Input layers in the final IR.

Input channels re-ordering, scaling, subtraction of mean values and other precprocessing features
are not applied by default. To pass necessary values to Model Optmizer, please run <code>mo.py</code>
(or <code>mo_tf.py</code>, <code>mo_caffe.py</code>, <code>mo_mxnet.py</code>) with <code>--help</code> and
examine all available options.

## Working with Inference Engine

To the moment, Inference Engine is the only consumer of IR models that Model Optimizer produces.
The whole workflow and more documentation on the structure of IR are documented in the Developer Guide
of Inference Engine. Note that sections about running Model Optimizer refer to the old version
of the tool and can not be applied to the current version of Model Optimizer.


## Setup development environment

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
    pylint mo/ mo.py
</pre>

### How to check requirements dependencies 

1. Run the following command:
<pre>
    safety check -r requirements_file
</pre>

> **NOTE**: here <code>requirements_file</code> is one of the following: <code>requirements.txt</code>, <code>requirements_caffe.txt</code>, <code>requirements_tf.txt</code>, <code>requirements_mxnet.txt</code>, <code>requirements_dev.txt</code>.
