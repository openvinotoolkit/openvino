## Prerequisites

Model Optimizer requires:

1. Python 3 or newer

2. [Optional] Please read about use cases that require Caffe\* to be available on the machine in the documentation.

## Installation instructions

1. Go to the Model Optimizer folder:
<pre>
    cd PATH_TO_INSTALL_DIR/deployment_tools/model_optimizer
</pre>

2. Create virtual environment and activate it. This option is strongly recommended as it creates a Python sandbox and
   dependencies for the Model Optimizer do not influence global Python configuration, installed libraries etc. At the
   same time, special flag ensures that system-wide Python libraries are also available in this sandbox. Skip this
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
   available <code>requirements_\*.txt</code> files corresponding to the framework of choice. For example, for Caffe
   use <code>requirements_caffe.txt</code> and so on. When you decide to switch later to other frameworks, please
   install dependencies for them using the same mechanism:
   <pre>
   pip3 install -r requirements.txt
   </pre>
   Or you can use the installation scripts from the "install_prerequisites" directory.

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
    pylint mo/ extensions/ mo.py
</pre>

