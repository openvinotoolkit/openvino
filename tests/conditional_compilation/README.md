# Conditional compilation tests

This folder contains conditional compilation (CC) test framework code and CC tests file.

## Environment preparation:
Install Python modules required for tests:
```bash
pip3 install -r requirements.txt 
```

## Run tests

```bash
pytest test_cc.py
```
**Test parameters:**
- `sea_runtool` - path to `sea_runtool.py` file.
- `collector_dir` - path to collector file parent folder.
- `artifacts` - Path to directory where test write output or read input.
- `openvino_root_dir` - Path to OpenVINO repo root directory.

**Optional:**
- `test_conf` - path to test cases .yml config.
- `openvino_ref` - Path to root directory with installed OpenVINO. If the option is not specified, CC test firstly build and install
    instrumented package at `<artifacts>/ref_pkg` folder with OpenVINO repository specified in `--openvino_root_dir` option.
    > If OpenVINO instrumented package has been successfuly installed, in the future you can set `--openvino_ref` parameter as `<artifacts>/ref_pkg` for better performance.

**Sample usage:**
```bash
pytest test_cc.py --sea_runtool=./thirdparty/itt_collector/runtool/sea_runtool.py --collector_dir=./bin/intel64/Release --artifacts=../artifacts --openvino_root_dir=.
```
