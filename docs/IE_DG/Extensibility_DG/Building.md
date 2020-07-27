# Build Extension Library Using CMake* {#openvino_docs_IE_DG_Extensibility_DG_Building}

Inference Engine build infrastructure provides the Inference Engine Package for application development.

To build an extension library, use the following CMake script:

@snippet CMakeLists.txt cmake:extension

This CMake script finds the Inference Engine and nGraph using the `find_package` CMake command.

To build an extension library, run the commands below:

```sh
$ cd template_extension
$ mkdir build
$ cd build
$ cmake -DInferenceEngine_DIR=[IE_DIR] -Dngraph_DIR=[NGRAPH_DIR] ../
$ cmake --build .
```
