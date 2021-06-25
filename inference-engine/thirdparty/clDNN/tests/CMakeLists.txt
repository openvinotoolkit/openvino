# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ========================================= Name / Output settings =====================================

set(CLDNN_BUILD__PROJ             "clDNN_unit_tests")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# =========================================== Compiler options =========================================
intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET
      WarnLevel3
      StandardCxx11
  )
if (NOT MSVC)
  intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
      SET_RAW
        "-Wno-error=conversion-null"
        "-Wno-error=type-limits"
    )
endif ()

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

set(__CLDNN_Directory__test_cases      "${CMAKE_CURRENT_SOURCE_DIR}/test_cases")
set(__CLDNN_Label__test_cases          "test cases")
file(GLOB __CLDNN_Sources__test_cases
    "${__CLDNN_Directory__test_cases}/*.h"
    "${__CLDNN_Directory__test_cases}/*.hpp"
    "${__CLDNN_Directory__test_cases}/*.cpp"
  )

set(__CLDNN_Directory__test_utils      "${CMAKE_CURRENT_SOURCE_DIR}/test_utils")
set(__CLDNN_Label__test_utils          "test utils")
file(GLOB __CLDNN_Sources__test_utils
    "${__CLDNN_Directory__test_utils}/*.h"
    "${__CLDNN_Directory__test_utils}/*.hpp"
    "${__CLDNN_Directory__test_utils}/*.cpp"
  )

set(__CLDNN_Directory__module_tests      "${CMAKE_CURRENT_SOURCE_DIR}/module_tests")
set(__CLDNN_Label__module_tests          "module tests")
file(GLOB __CLDNN_Sources__module_tests
    "${__CLDNN_Directory__module_tests}/*.h"
    "${__CLDNN_Directory__module_tests}/*.hpp"
    "${__CLDNN_Directory__module_tests}/*.cpp"
  )

if (MSVC)
  file(GLOB __CLDNN_Sources__natvis
    "${CMAKE_CURRENT_SOURCE_DIR}/float16.natvis"
  )
endif()

set(__CLDNN_Directory__ks_main           "${CLDNN__KERNEL_SELECTOR_DIR}")
set(__CLDNN_Directory__ks_core           "${CLDNN__KERNEL_SELECTOR_DIR}/core")
set(__CLDNN_Directory__ks_common         "${CLDNN__KERNEL_SELECTOR_DIR}/common")
set(__CLDNN_Directory__ks_core_common    "${__CLDNN_Directory__ks_core}/common")
set(__CLDNN_Directory__ks_actual_kernels "${__CLDNN_Directory__ks_core}/actual_kernels")
set(__CLDNN_Directory__ks_cache          "${__CLDNN_Directory__ks_core}/cache")

set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__test_cases}
    ${__CLDNN_Sources__test_utils}
    ${__CLDNN_Sources__module_tests}
    ${__CLDNN_Sources__natvis}
  )

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"         FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__test_cases}"   FILES ${__CLDNN_Sources__test_cases})
source_group("${__CLDNN_Label__test_utils}"   FILES ${__CLDNN_Sources__test_utils})
source_group("${__CLDNN_Label__module_tests}" FILES ${__CLDNN_Sources__module_tests})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CLDNN__MAIN_DIR}/src"
    "${CLDNN__MAIN_DIR}/src/include/"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLDNN__RUNTIME_DIR}"
    "${__CLDNN_Directory__test_utils}"
    "${__CLDNN_Directory__ks_core}"
    "${__CLDNN_Directory__ks_core}/common"
    "${__CLDNN_Directory__ks_actual_kernels}"
    "${__CLDNN_Directory__ks_common}"
  )

# =================================== Link targets and dependencies ====================================

# Tests executable.
add_executable("${CLDNN_BUILD__PROJ}"
    ${__CLDNN_AllSources}
  )

if(COMMAND set_ie_threading_interface_for)
  set_ie_threading_interface_for("${CLDNN_BUILD__PROJ}")
endif()

set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")

# Set library dependencies
target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE "${CLDNN_BUILD__PROJ__clDNN}" OpenCL gtest gmock)

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE pthread)
endif()
target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE ${CLDNN__SYSTEM_LINK_LIBRARIES})

# =================================== Custom pre- and post-steps =======================================

if(CLDNN__RUN_TESTS)
  add_custom_command(TARGET "${CLDNN_BUILD__PROJ}" POST_BUILD
      WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
      COMMAND "${CLDNN_BUILD__PROJ}"
      COMMENT "Executing tests..."
    )
endif()

# ======================================================================================================
