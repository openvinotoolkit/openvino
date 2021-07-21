# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Name of project (helper constant variable).
set(CLDNN__PROJ_NAME "clDNN")

# Default languages: C, C++.
project("${CLDNN__PROJ_NAME}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ======================================================================================================
# ====================================== HELPER CONSTANT VARIABLES =====================================
# ======================================================================================================
# ======================================================================================================

if(ENABLE_GPU_DEBUG_CAPS)
  add_definitions(-DGPU_DEBUG_CONFIG=1)
endif()

# Path which points to main directory of project.
set(CLDNN__MAIN_DIR      "${CMAKE_CURRENT_SOURCE_DIR}")

# Path which points to directory with interface for framework.
set(CLDNN__API_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/api")

# Path which points to directory with interface for framework.
set(CLDNN__KERNEL_SELECTOR_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/kernel_selector")

# Path which points to directory with runtime
set(CLDNN__RUNTIME_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/runtime")

set(GPU_BACKEND "L0" CACHE STRING "selected GPU backed (OCL/L0)" FORCE)
find_package(LevelZero REQUIRED)

# Code generation settings.

# Path which points to root directory where code generated elements are created.
set(CLDNN__CODEGEN_BASEDIR "${CMAKE_CURRENT_BINARY_DIR}/codegen")
# Path which points to root directory where code generated elements are created
# (specific to build configuration).
set(CLDNN__CODEGEN_DIR     "${CLDNN__CODEGEN_BASEDIR}")

# Path which points to automatically included directory with code generated elements
# (to support "copy-if-different" optimization).
set(CLDNN__CODEGEN_INCDIR  "${CLDNN__CODEGEN_DIR}/include")

# ======================================================================================================
# ============================================ CMAKE OPTIONS ===========================================
# ======================================================================================================

# Include and build: Tests (unit tests and small acceptance tests) for clDNN framework.
set(CLDNN__INCLUDE_TESTS ON CACHE BOOL "Include and build: clDNN framework's tests.")
mark_as_advanced(CLDNN__INCLUDE_TESTS)

# ======================================================================================================

# Check for python 3 interpreter (required tool).
find_package(PythonInterp 3 QUIET)
if(NOT PYTHONINTERP_FOUND)
  message(WARNING "[clDNN] Project requires Python 3.x interpreter to build (with python loader). CMake could not detect it correctly.
      If you have installed this interpreter, please disregard this warning or specify PYTHON_EXECUTABLE in CMake command-line."
    )
endif()

# ======================================================================================================
set(CLDNN_UTILS__RAPIDJSON_INCDIRS "utils/rapidjson" CACHE INTERNAL "Paths to interface headers for rapidjson.")

# ======================================================================================================
# ==================================== COMMON BUILD CONFIGURATION ======================================
# ======================================================================================================

# =================================== Main targets names and labels ====================================

set(CLDNN_BUILD__PROJ__clDNN       "clDNN_lib")
set(CLDNN_BUILD__PROJ_LABEL__clDNN "clDNN")

# ===================================== Include/Link directories =======================================

include_directories(
    ${CLDNN_UTILS__RAPIDJSON_INCDIRS}
    "${CLDNN__CODEGEN_INCDIR}"
    "${CLDNN__API_DIR}"
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
  )

# =================================== Link targets and dependencies ====================================
add_subdirectory(src)
add_subdirectory(runtime)

if(CLDNN__INCLUDE_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(kernel_selector)

target_link_libraries(${CLDNN_BUILD__PROJ__clDNN} PRIVATE inference_engine)