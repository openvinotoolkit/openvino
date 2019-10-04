# Copyright (c) 2016 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        "-Wno-error=unused-variable"
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

set(__CLDNN_Directory__gtest           "${CLDNN__GTEST_DIR}")
set(__CLDNN_Label__gtest               "google test framework")
file(GLOB __CLDNN_Sources__gtest
    "${__CLDNN_Directory__gtest}/*.cc"
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
    ${__CLDNN_Sources__gtest}
    ${__CLDNN_Sources__natvis}
  )

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"         FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__test_cases}"   FILES ${__CLDNN_Sources__test_cases})
source_group("${__CLDNN_Label__test_utils}"   FILES ${__CLDNN_Sources__test_utils})
source_group("${__CLDNN_Label__module_tests}" FILES ${__CLDNN_Sources__module_tests})
source_group("${__CLDNN_Label__gtest}"        FILES ${__CLDNN_Sources__gtest})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CLDNN__MAIN_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${__CLDNN_Directory__test_utils}"
    "${CLDNN__GTEST_DIR}"
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
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")

# Set library dependencies
target_link_libraries("${CLDNN_BUILD__PROJ}"
    "${CLDNN_BUILD__PROJ__clDNN}"
  )

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries("${CLDNN_BUILD__PROJ}" pthread)
endif()
target_link_libraries("${CLDNN_BUILD__PROJ}" ${CLDNN__SYSTEM_LINK_LIBRARIES})

# =================================== Custom pre- and post-steps =======================================

if(CLDNN__RUN_TESTS)
  add_custom_command(TARGET "${CLDNN_BUILD__PROJ}" POST_BUILD
      WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
      COMMAND "${CLDNN_BUILD__PROJ}"
      COMMENT "Executing tests..."
    )
endif()

# ======================================================================================================
