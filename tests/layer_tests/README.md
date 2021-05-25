## How to setup

* Install requirements:

> pip3 install -r requirements.txt

* Set up environment variables for layer tests:

> export MO_ROOT=PATH_TO_MO

> export PYTHONPATH="path_to_openvino"/tests/layer_tests/:$PYTHONPATH

* To compare scoring results need:
    
    * Set up additional environment variables:
    
        >export IE_APP_PATH="path_to_IE"
    
        >export IE_CUSTOM_LAYER="path_to_IE"
    
    * Add IE dependencies in LD_LIBRARY_PATH 
    
    * To use timelimit tool add path to the tool in PATH
    
    * To use python_api add path to python_api in PYTHONPATH 

* Run layer tests:

> py.test
