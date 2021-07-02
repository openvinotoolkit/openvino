# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.1)
include(CheckCXXCompilerFlag)

# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 1) ===================================
# ======================================================================================================
# The section must be before project definition which is a point for configurations generation
# for multi-configuration generators.

# Available configuration types:
# Two standard configurations.
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES
        "Debug"
        "Release"
        "RelWithDebInfo"
      )
endif()

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

# Name of project (helper constant variable).
set(CLDNN__PROJ_NAME "clDNN")

# Default languages: C, C++.
project("${CLDNN__PROJ_NAME}")

if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 2) ===================================
# ======================================================================================================
# The section must be after project definition and its supported languages because otherwise
# used variables are not available yet.

# Populating global property with list of debug configurations.
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS "Debug")
# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# ======================================================================================================
# ====================================== HELPER CONSTANT VARIABLES =====================================
# ======================================================================================================
# ======================================================================================================
if(CLDNN_THREADING MATCHES "SEQ")
    add_definitions(-DCLDNN_THREADING=CLDNN_THREADING_SEQ)
elseif(CLDNN_THREADING MATCHES "TBB")
    add_definitions(-DCLDNN_THREADING=CLDNN_THREADING_TBB)
else()
    add_definitions(-DCLDNN_THREADING=CLDNN_THREADING_THREADPOOL)
endif()

if(GPU_DEBUG_CONFIG)
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

# Include and build: Core of clDNN framework.
set(CLDNN__INCLUDE_CORE ON CACHE BOOL "Include and build: clDNN core.")
mark_as_advanced(CLDNN__INCLUDE_CORE)

# ======================================================================================================

# Include and build: Kernel selector for clDNN framework.
set(CLDNN__INCLUDE_KERNEL_SELECTOR ON CACHE BOOL "Include and build: clDNN kernel selector.")
mark_as_advanced(CLDNN__INCLUDE_KERNEL_SELECTOR)

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

# ================================================ Outputs =============================================

set(CLDNN_BUILD__PROJ_OUTPUT_NAME__clDNN "clDNN${CLDNN__OUT_CPU_SUFFIX}")

# ===================================== Include/Link directories =======================================

include_directories(
    ${CLDNN_UTILS__RAPIDJSON_INCDIRS}
    "${CLDNN__CODEGEN_INCDIR}"
    "${CLDNN__API_DIR}"
  )

# =================================== Link targets and dependencies ====================================
if(CLDNN__INCLUDE_CORE)
  add_subdirectory(src)
  add_subdirectory(runtime)
endif()
if(CLDNN__INCLUDE_TESTS)
  add_subdirectory(tests)
endif()
if(CLDNN__INCLUDE_KERNEL_SELECTOR)
  add_subdirectory(kernel_selector)
endif()
