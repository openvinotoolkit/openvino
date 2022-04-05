These tests execute IE samples on pregenerated IR

<INSTALL_DIR> - OpenVINO install directory

You can run tests not only from the <INSTALL_DIR>, but in this case you need to remember to adjust the environment variables like as WORKSPACE and SHARE

To install smoke tests:
    ``` bash                                            			
    - cd <working directory>/tests/samples_tests/smoke_tests
    - mkdir build && cd build
    - cmake ../..
    - cmake -DCOMPONENT=tests -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -P cmake_install.cmake
    ```
1. To run tests from install directory required installing some dependencies: 
    - pip3 install -r  <INSTALL_DIR>\tests\smoke_tests\requirements.txt
2. Call setupvars script and then set the environment variables:
a. Required:
    - IE_APP_PATH : coomon path to C++ and C samples, e.g. '<INSTALL_DIR>/samples_bin'
    - IE_APP_PYTHON_PATH : path to python IE samples, e.g. '<INSTALL_DIR>/samples/python/'
    - IE_APP_PYTHON_TOOL_PATH : path to python IE tools for benchmark_app, e.g. '<INSTALL_DIR>/tools/' 
b. Optional:
    - TEST_DEVICE = CPU by default
3. Configure env_config.yml according to your paths:
    - Set WORKSPACE : working directory, e.g. '<INSTALL_DIR>'
    - Set SHARE : path to loaded data with models, e.g. '<INSTALL_DIR>/tests/smoke_tests/samples_smoke_tests_data/' 
4. Run all test via pytest:	
    - python -m pytest --env_conf env_config.yml -s 
5. Run only one sample (for example, classification_sample_async):
    - python -m pytest test_classification_sample_async.py  --env_conf env_config.yml -s 
6. To run performance add pytest key: "performance n", where n is number of perf iteration.
   Test finds in output of sample 'fps', if it exists,
   then tests rerun that sample adding key 'niter n' with number of perfomance run (that you passed to pytest with '--performance n' keys)
   Not to add 'niter' key, please, execute pytest "--performance 0"

This test using pregenerated IRs, that located right now in shared folder mentioned above. Also data (images, videos and others) locates in that shared folder.
