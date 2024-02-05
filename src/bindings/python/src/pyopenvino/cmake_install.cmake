# Install script for directory: D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/OpenVINO")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino/frontend/onnx/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino/frontend/tensorflow/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino/frontend/paddle/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino/frontend/pytorch/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "pyopenvino_python3.11" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/src/bindings/python/src/openvino" USE_SOURCE_PERMISSIONS REGEX "/test\\_utils$" EXCLUDE REGEX "/torchvision\\/requirements\\.txt$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "pyopenvino_python3.11" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/openvino" TYPE MODULE FILES "D:/OPENVINO/openvino/bin/intel64/Debug/python/openvino/_pyopenvinod.cp311-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/openvino" TYPE MODULE FILES "D:/OPENVINO/openvino/bin/intel64/Release/python/openvino/_pyopenvino.cp311-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/openvino" TYPE MODULE FILES "D:/OPENVINO/openvino/bin/intel64/MinSizeRel/python/openvino/_pyopenvino.cp311-win_amd64.pyd")
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/openvino" TYPE MODULE FILES "D:/OPENVINO/openvino/bin/intel64/RelWithDebInfo/python/openvino/_pyopenvino.cp311-win_amd64.pyd")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "pyopenvino_python3.11" OR NOT CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    include("D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino/CMakeFiles/pyopenvino.dir/install-cxx-module-bmi-Debug.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    include("D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino/CMakeFiles/pyopenvino.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    include("D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino/CMakeFiles/pyopenvino.dir/install-cxx-module-bmi-MinSizeRel.cmake" OPTIONAL)
  elseif(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    include("D:/OPENVINO/openvino/src/bindings/python/src/pyopenvino/CMakeFiles/pyopenvino.dir/install-cxx-module-bmi-RelWithDebInfo.cmake" OPTIONAL)
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openvino_req_files" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES "D:/OPENVINO/openvino/src/bindings/python/requirements.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openvino_req_files" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/openvino/preprocess/torchvision" TYPE FILE FILES "D:/OPENVINO/openvino/src/bindings/python/src/openvino/preprocess/torchvision/requirements.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "tests")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/tests/pyopenvino" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/src/bindings/python/tests")
endif()

