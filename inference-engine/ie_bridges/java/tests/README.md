## Build
Create an environment variable with Inference Engine installation path:
```bash
export IE_PATH=/path/to/openvino/bin/intel64/Release/lib/
```

To create java wrappers tests add `-DENABLE_JAVA=ON` and `-DENABLE_TESTS=ON` flags in cmake command while building openvino:
```bash
cd openvino
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_JAVA=ON -DENABLE_TESTS=ON ..
make --jobs=$(nproc --all)
```

Link to repo with [test data](https://github.com/openvinotoolkit/testdata.git)

## Running
Create an environment variable with testdata path:
```bash
export MODELS_PATH=/path/to/testdata
```

Add library path for openvino java library before running:
```bash
export LD_LIBRARY_PATH=${IE_PATH}:$LD_LIBRARY_PATH
```

To run tests use:
```bash
java -cp ".:${IE_PATH}/*" OpenVinoTestRunner
```
