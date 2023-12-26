# CPU Dump Check Tool

# Preparing

 1. Install dependencies:

```bash
pip3 install -r ./requirements.txt
```

 2. Build CPU plugin with `-DENABLE_DEBUG_CAPS=ON` and install it.

 3. Initialize OpenVINO enviroment:
 
 ```bash
 # suppose CMAKE_INSTALL_PREFIX=~/openvino/build/install
 source ~/openvino/build/install/setupvars.sh
 ```
 # Typical usage
 
 - dump each output tensors from CPU plugin:
```bash
python3 dump_check.py -m=/path/to/model dump1
```

 - comparing two dumps and analyze differences:
```bash
python3 dump_check.py -m=/path/to/model dump1 dump2
```

 - visualize first error map:
```bash
python3 dump_check.py -m=/path/to/model dump1 dump2 -v
```

more options can be learned from the help of this tool.

 # Compare BF16 dump with FP32 reference
CPU plugin would dump BF16 blob as int16_t type buffer.Dumping all the nodes
and compare usually would fail, because BF16 executable graph differs with fp32.
However, we can dump some computing intensive node output and compare result error
following the steps:
```bash
# step 1: dump bf16 precision
python3 dump_check.py -m /path/model.xml -bf16 -f Convolution ./dump_bf16

# step 2: dump fp32 precision
python3 dump_check.py -m /path/model.xml  -f Convolution ./dump_fp32

# step 3: compare precision between bf16 and fp32
python3 dump_check.py -m /path/model.xml  ./dump_bf16 ./dump_fp32
```
