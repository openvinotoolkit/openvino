These test executes IE samples on pregenerated IR

1. To run tests required installing some dependencies: 
    - pip3 install -r requirements.txt

2. Download archive with models and validation set: [samples tests data](https://storage.openvinotoolkit.org/repositories/openvino/ci_dependencies/test/samples_smoke_tests_data.zip)
3. Set environment variables:
a. Required:
```sh
- IE_APP_PATH=<path to samples only for C++ and C samples>
- IE_APP_PYTHON_PATH=<path to python IE samples only for python samples>
- PYTHONPATH=<path to samples_smoke_tests>$PYTHONPATH>
- LD_LIBRARY_PATH=including all plugins
- IE_APP_PYTHON_TOOL_PATH=<path to python IE tools for benchmark_app>
```	
b. Optional:
	- TEST_DEVICE = CPU by default
4. Configure env_config.yml according to your paths:
    - Set WORKSPACE (your working directory) and SHARE - path to share 
5. Run all test via pytest:	
    - pytest --env_conf env_config.yml -s 
6. Run only one sample (for example, classification_sample_async):
    - pytest test_classification_sample_async.py  --env_conf env_config.yml -s 
7. To run performance add pytest key: "performance n", where n is number of perf iteration.
   Test finds in output of sample 'fps', if it exists,
   then tests rerun that sample adding key 'niter n' with number of perfomance run (that you passed to pytest with '--performance n' keys)
   Not to add 'niter' key, please, execute pytest "--performance 0"

This test using pregenerated IRs, that located right now in shared folder mentioned above. Also data (images, videos and others) locates in that shared folder.
