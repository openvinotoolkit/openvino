# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_GPU_DEBUG_CAPS)
  add_definitions(-DGPU_DEBUG_CONFIG=1)
endif()

set(MAIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/api")

find_package(PythonInterp 3 QUIET)
if(NOT PYTHONINTERP_FOUND)
  message(WARNING "[clDNN] Project requires Python 3.x interpreter to build (with python loader). CMake could not detect it correctly.
      If you have installed this interpreter, please disregard this warning or specify PYTHON_EXECUTABLE in CMake command-line."
    )
endif()

add_subdirectory(runtime)
add_subdirectory(kernel_selector)
add_subdirectory(src)
add_subdirectory(utils)

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()
