# Conditional compilation tests

This folder contains conditional compilation (CC) test framework code and CC tests file

## Run tests
    ```
    pytest test_cc.py
    ```
    `test_cc.py` options

        --sea_runtool=./thirdparty/itt_collector/runtool/sea_runtool.py \
    --collector_dir=./bin/intel64/Release \
    --artifacts=<path to directory where tests write output and read input> \
    --openvino_ref=<path to root directory with installed OpenVINO> \
    --openvino_root_dir=<path to OpenVINO repo root directory>
    
    --sea_runtool=./thirdparty/itt_collector/runtool/sea_runtool.py \
    --collector_dir=./bin/intel64/Release \
    --artifacts=<path to directory where tests write output and read input> \
    --openvino_ref=<path to root directory with installed OpenVINO> \
    --openvino_root_dir=<path to OpenVINO repo root directory>