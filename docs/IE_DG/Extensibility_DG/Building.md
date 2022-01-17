# Build Extension Library Using CMake* {#openvino_docs_IE_DG_Extensibility_DG_Building}

Inference Engine build infrastructure provides the Inference Engine Package for application development.

To configure the build of your extension library, use the following CMake script:

@snippet template_extension/old/CMakeLists.txt cmake:extension

This CMake script finds the Inference Engine and nGraph using the `find_package` CMake command.

To build the extension library, run the commands below:

```sh
$ cd template_extension/old
$ mkdir build
$ cd build
$ cmake -DOpenVINO_DIR=[OpenVINO_DIR]  ../
$ cmake --build .
```
