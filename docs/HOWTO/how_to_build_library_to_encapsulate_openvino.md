# How to build library to encapsulate OpenVINO

> **NOTE**: This guide illustrated a simplified example about how to build a customer library based on OpenVINO.

## A simplifiled library source code

* The source structure
```
├── library-example
│   ├── CMakeLists.txt
│   ├── include
│   │   └── library-example.h
│   └── src
│       └── library-example.cpp
```

* `CMakeLists.txt`
```sh
set(TARGET_NAME "customer-library")

project(TARGET_NAME)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(${TARGET_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
```
> **NOTE**: Make sure the OpenVINO package can be found: `find_package(OpenVINO REQUIRED COMPONENTS Runtime)`.
> **NOTE**: Make customer library link to OpenVINO library: `target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)`.

* `library-example.h`
```sh
void print_ov_version();
```

* `library-example.cpp`
```sh
#include "../include/library-example.h"
#include "openvino/openvino.hpp"
void print_ov_version() {
    ov::Version object = ov::get_openvino_version();
    std::cout << "ov version: " << object.buildNumber << std::endl;
    return;
}
```
> **NOTE**: Simple example call `ov::get_openvino_version()` function from OpenVINO.


## Build the library

* enable the ENV of OpenVINO (Ubuntu)
```sh
$ cd path_to/openvino_package
$ source setupvars.sh
```

* run cmake
```sh
$ cd library-example
$ mkdir build && cd build
$ cmake ..
```

* build to generate the libcustomer-library.so
```sh
$ make
$ ll | grep lib
-rwxrwxr-x 1 odt odt 39744  七  28 13:28 libcustomer-library.so*
```

* check the library link
```sh
$ ldd libcustomer-library.so | grep openvino
libopenvino.so => ~/openvino-install-dir/runtime/lib/intel64/libopenvino.so (0x00007f42b4f9a000)
```
## Test the library
* A simple test binary
```sh
#include "include/library-example.h"

int main(int argc, char** argv) {
    print_ov_version();
    return 0;
}
```

* compiler the sample
```sh
g++ -o mytest main.cpp -L. -lcustomer-library
```

* run the sample
```sh
./TEST
ov version: xxxxxxxx_164a59925abf939094ac068e3fdd373729ba60c8
```

