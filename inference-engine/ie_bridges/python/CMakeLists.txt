# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Defines the CMake commands/policies
cmake_minimum_required( VERSION 2.8.5 )

# Set the project name
project( INFERENCE_ENGINE_DRIVER )

option(COPY_IE_LIBS "Copy Inference Engine libs to package directory" ${WIN32})

set (IE_DEFAULT_PATH computer_vision_sdk/deployment_tools/inference_engine/share)

find_package(InferenceEngine REQUIRED PATHS /opt/intel/${IE_DEFAULT_PATH} $ENV{HOME}/intel/${IE_DEFAULT_PATH})

link_directories(${LIB_FOLDER})

# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake )

# Include the CMake script UseCython.cmake.  This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include( UseCython )

# With CMake, a clean separation can be made between the source tree and the
# build tree.  When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree.  However, with pure *.py source, the
# source is processed directly.  To handle this, we reproduce the availability
# of the source files in the build tree.
add_custom_target( ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )

add_custom_target( CopyIeLibs ${CMAKE_COMMAND} -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyIeLibs.cmake
  ${IE_ROOT_DIR}/bin/${_ARCH}/Release ${_IE_ROOT_LIBRARY}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ie_driver )

if(NOT(UNIX))
    include_directories( IE::inference_engine )
else()
    include_directories( ${IE_MAIN_SOURCE_DIR}/include )
endif()

# Process the CMakeLists.txt in the 'src' and 'bin' directory.
add_subdirectory( inference_engine )
