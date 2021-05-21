# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ========================================= Name / Output settings =====================================

set(CLDNN_BUILD__PROJ             "clDNN_tests_core_internal")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# =========================================== Compiler options =========================================
intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET
	  StandardCxx11
      RttiEnabled
  )

if (NOT MSVC)
  intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
      SET_RAW
        "-Wno-error=conversion-null"
        "-Wno-error=type-limits"
    )
endif ()

find_package(OpenMP)
if (OPENMP_FOUND)
    add_definitions(-DOPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ================================== Compiler preprocessor definitions =================================

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

set(__CLDNN_Directory__test_utils      "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_utils")
set(__CLDNN_Label__test_utils          "test utils")
file(GLOB __CLDNN_Sources__test_utils
    "${__CLDNN_Directory__test_utils}/*.h"
    "${__CLDNN_Directory__test_utils}/*.hpp"
    "${__CLDNN_Directory__test_utils}/*.cpp"
  )

set(__CLDNN_Directory__gtest           "${CLDNN__GTEST_DIR}")
set(__CLDNN_Label__gtest               "google test framework")
file(GLOB __CLDNN_Sources__gtest
    "${__CLDNN_Directory__gtest}/*.cc"
  )

if(UNIX)
    set_source_files_properties(${__CLDNN_Sources__gtest} PROPERTIES
        COMPILE_FLAGS "-Wno-undef")
endif()

set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__test_cases}
    ${__CLDNN_Sources__test_utils}
    ${__CLDNN_Sources__gtest}
  )
# Helping with some generators.
set_property(SOURCE ${__CLDNN_Sources__cg_cache} PROPERTY GENERATED TRUE)


# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"                 FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__test_cases}"           FILES ${__CLDNN_Sources__test_cases})
source_group("${__CLDNN_Label__test_utils}"           FILES ${__CLDNN_Sources__test_utils})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CLDNN__MAIN_DIR}/src"
    "${CLDNN__MAIN_DIR}/src/include"
    "${CLDNN__KERNEL_SELECTOR_DIR}/core"
    "${CLDNN__KERNEL_SELECTOR_DIR}/core/common"
    "${CLDNN__KERNEL_SELECTOR_DIR}/common"
    "${__CLDNN_Directory__test_utils}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )

include_directories(SYSTEM
    "${CLDNN__GTEST_DIR}")

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
target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE "${CLDNN_BUILD__PROJ__clDNN}")

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
