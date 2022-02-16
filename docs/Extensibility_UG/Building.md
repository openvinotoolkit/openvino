# Build Extension Library Using CMake* {#openvino_docs_IE_DG_Extensibility_UG_Building}

OpenVINO™ Runtime build infrastructure provides the OpenVINO™ Package for application development.

To configure the build of your extension library, use the following CMake script:

@snippet template_extension/new/CMakeLists.txt cmake:extension

This CMake script finds the OpenVINO™ using the `find_package` CMake command.

To build the extension library, run the commands below:

```sh
$ cd docs/template_extension/new
$ mkdir build
$ cd build
$ cmake -DOpenVINO_DIR=[OpenVINO_DIR] ../
$ cmake --build .
```
