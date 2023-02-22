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

