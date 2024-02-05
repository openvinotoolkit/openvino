# Install script for directory: D:/OPENVINO/openvino/samples

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
  include("D:/OPENVINO/openvino/samples/cpp/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("D:/OPENVINO/openvino/samples/c/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cpp_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/cpp" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/samples/cpp/" REGEX "/[^/]*\\.sh$" EXCLUDE REGEX "/\\.clang\\-format$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "c_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/c" TYPE PROGRAM FILES
    "D:/OPENVINO/openvino/samples/cpp/build_samples_msvc.bat"
    "D:/OPENVINO/openvino/samples/cpp/build_samples.ps1"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "c_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/samples/c" REGEX "/c\\/cmakelists\\.txt$" EXCLUDE REGEX "/c\\/\\.clang\\-format$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "c_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/c" TYPE FILE FILES "D:/OPENVINO/openvino/samples/cpp/CMakeLists.txt")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "python_samples" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/samples/python" TYPE DIRECTORY FILES "D:/OPENVINO/openvino/samples/python/")
endif()

