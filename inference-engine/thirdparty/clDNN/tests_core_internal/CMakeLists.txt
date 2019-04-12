# Copyright (c) 2019 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ====================================== Helper constant variables =====================================

# Order of scan for special capabilities files (.inc files with capabilities description).
set(CLDNN__CAPS_SCAN_ORDER
    "private"
    "internal"
    "public"
  )

# ========================================= Name / Output settings =====================================

set(CLDNN_BUILD__PROJ             "tests_core_internal")
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
        "-Wno-error=unused-variable"
    )
endif ()

find_package(OpenMP)
if (OPENMP_FOUND)
    add_definitions(-DOPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ================================== Compiler preprocessor definitions =================================

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    CLDNN_EXPORTS
    EXPORT_NEURAL_SYMBOLS
    "CLDNN_VERSION_MAJOR=${CLDNN__VERSION_MAJOR}"
    "CLDNN_VERSION_MINOR=${CLDNN__VERSION_MINOR}"
    "CLDNN_VERSION_BUILD=${CLDNN__VERSION_BUILD}"
    "CLDNN_VERSION_REVISION=${CLDNN__VERSION_REVISION}"
  )


# ========================================= Source/Header files ========================================

set(__CLDNN_Directory__clDNN_copy             "${CMAKE_CURRENT_SOURCE_DIR}/../src")
set(__CLDNN_Label__clDNN_copy                 "clDNN")
file(GLOB __CLDNN_Sources__clDNN_copy
    "${__CLDNN_Directory__clDNN_copy}/*.h"
    "${__CLDNN_Directory__clDNN_copy}/*.hpp"
    "${__CLDNN_Directory__clDNN_copy}/*.cpp"
    "${__CLDNN_Directory__clDNN_copy}/*.inc"
  )

set(__CLDNN_Label__api                 "${__CLDNN_Label__clDNN_copy}\\api")
file(GLOB __CLDNN_Headers__api
    "${CLDNN__API_DIR}/*.h"
    "${CLDNN__API_DIR}/*.hpp"
  )

set(__CLDNN_Directory__api__cpp "${CLDNN__API_DIR}/CPP")
set(__CLDNN_Label__api__cpp     "${__CLDNN_Label__api}\\CPP")
file(GLOB __CLDNN_Headers__api__cpp
    "${__CLDNN_Directory__api__cpp}/*.h"
    "${__CLDNN_Directory__api__cpp}/*.hpp"
  )

set(__CLDNN_Directory__api__c "${CLDNN__API_DIR}/C")
set(__CLDNN_Label__api__c     "${__CLDNN_Label__api}\\C")
file(GLOB __CLDNN_Headers__api__c
    "${__CLDNN_Directory__api__c}/*.h"
    "${__CLDNN_Directory__api__c}/*.hpp"
  )

set(__CLDNN_Label__api_extension       "${__CLDNN_Label__clDNN_copy}\\api_extension")
file(GLOB __CLDNN_Headers__api_extension
    "${CLDNN__API_EXTENSION_DIR}/*.h"
    "${CLDNN__API_EXTENSION_DIR}/*.hpp"
  )

set(__CLDNN_Directory__api_extension__cpp "${CLDNN__API_EXTENSION_DIR}/CPP")
set(__CLDNN_Label__api_extension__cpp     "${__CLDNN_Label__api_extension}\\CPP")
file(GLOB __CLDNN_Headers__api_extension__cpp
    "${__CLDNN_Directory__api_extension__cpp}/*.h"
    "${__CLDNN_Directory__api_extension__cpp}/*.hpp"
  )

set(__CLDNN_Directory__api_extension__c "${CLDNN__API_EXTENSION_DIR}/C")
set(__CLDNN_Label__api_extension__c     "${__CLDNN_Label__api_extension}\\C")
file(GLOB __CLDNN_Headers__api_extension__c
    "${__CLDNN_Directory__api_extension__c}/*.h"
    "${__CLDNN_Directory__api_extension__c}/*.hpp"
  )

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

set(__CLDNN_Directory__graph_opt       "${CMAKE_CURRENT_SOURCE_DIR}/../src/graph_optimizer")
set(__CLDNN_Label__graph_opt           "${__CLDNN_Label__clDNN_copy}\\graph_optimizer")
file(GLOB __CLDNN_Sources__graph_opt
    "${__CLDNN_Directory__graph_opt}/*.h"
    "${__CLDNN_Directory__graph_opt}/*.hpp"
    "${__CLDNN_Directory__graph_opt}/*.cpp"
  )

set(__CLDNN_Directory__include         "${CMAKE_CURRENT_SOURCE_DIR}/../src/include")
set(__CLDNN_Label__include             "${__CLDNN_Label__clDNN_copy}\\include")
file(GLOB __CLDNN_Headers__include
    "${__CLDNN_Directory__include}/*.h"
    "${__CLDNN_Directory__include}/*.hpp"
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

# Special handling of capabilities files.
set(__CLDNN_Directory__caps            "${CMAKE_CURRENT_SOURCE_DIR}/../src/caps")
set(__CLDNN_Label__caps                "${__CLDNN_Label__clDNN_copy}\\caps")
foreach(__CLDNN_CapsScanDir ${CLDNN__CAPS_SCAN_ORDER})
  string(REPLACE ";" "\;" __CLDNN_CapsScanDir "${__CLDNN_CapsScanDir}") # [WA#1] Must escape ; again if occurred in item.
  file(GLOB __CLDNN_Sources__caps "${__CLDNN_Directory__caps}/${__CLDNN_CapsScanDir}/*.inc")
  list(LENGTH __CLDNN_Sources__caps __CLDNN_CapsScanDirFileCount)
  if(__CLDNN_CapsScanDirFileCount GREATER 0)
    set(__CLDNN_IncDirectory__caps "${__CLDNN_Directory__caps}/${__CLDNN_CapsScanDir}")
    message(STATUS "[clDNN] Selected capabilities: ${__CLDNN_CapsScanDir}")
    break()
  endif()
endforeach()
if(NOT (__CLDNN_CapsScanDirFileCount GREATER 0))
  message(FATAL_ERROR "[clDNN] Cannot locate any capabilities files in \"${__CLDNN_Directory__caps}\" subdirectories.")
endif()
unset(__CLDNN_CapsScanDir)
unset(__CLDNN_CapsScanDirFileCount)

set(__CLDNN_Directory__gpu             "${CMAKE_CURRENT_SOURCE_DIR}/../src/gpu")
set(__CLDNN_Label__gpu                 "${__CLDNN_Label__clDNN_copy}\\gpu")
file(GLOB __CLDNN_Sources__gpu
    "${__CLDNN_Directory__gpu}/*.h"
    "${__CLDNN_Directory__gpu}/*.hpp"
    "${__CLDNN_Directory__gpu}/*.cpp"
    "${__CLDNN_Directory__gpu}/*.inc"
  )

set(__CLDNN_Directory__cache           "${__CLDNN_Directory__gpu}/cache")
set(__CLDNN_Label__cache               "${__CLDNN_Label__gpu}\\cache")
file(GLOB __CLDNN_Sources__cache
    "${__CLDNN_Directory__cache}/*.h"
    "${__CLDNN_Directory__cache}/*.hpp"
    "${__CLDNN_Directory__cache}/*.cpp"
  )

set(__CLDNN_Directory__ch_kernels      "${__CLDNN_Directory__cache}/kernels")
set(__CLDNN_Label__ch_kernels          "${__CLDNN_Label__cache}\\kernels")
file(GLOB __CLDNN_Sources__ch_kernels
    "${__CLDNN_Directory__ch_kernels}/*.cl"
  )

set(__CLDNN_Directory__cg_cache        "${CLDNN__CODEGEN_INCDIR}")
set(__CLDNN_CGDirectory__cg_cache      "${CLDNN__CODEGEN_DIR}/cache")
set(__CLDNN_Label__cg_cache            "${__CLDNN_Label__cache}\\codegen")

set(__CLDNN_Directory__ks_main           "${CLDNN__KERNEL_SELECTOR_DIR}")
set(__CLDNN_Directory__ks_core           "${CLDNN__KERNEL_SELECTOR_DIR}/core")
set(__CLDNN_Directory__ks_common         "${CLDNN__KERNEL_SELECTOR_DIR}/common")
set(__CLDNN_Directory__ks_core_common    "${__CLDNN_Directory__ks_core}/common")
set(__CLDNN_Directory__ks_actual_kernels "${__CLDNN_Directory__ks_core}/actual_kernels")
set(__CLDNN_Directory__ks_cache          "${__CLDNN_Directory__ks_core}/cache")


set(__CLDNN_AllSources
    ${__CLDNN_Sources__clDNN_copy}
    ${__CLDNN_Headers__api}
    ${__CLDNN_Sources__graph_opt}
    ${__CLDNN_Headers__include}
    ${__CLDNN_Sources__caps}
    ${__CLDNN_Headers__api__cpp}
    ${__CLDNN_Headers__api__c}
    ${__CLDNN_Headers__api_extension}
    ${__CLDNN_Headers__api_extension__c}
    ${__CLDNN_Headers__api_extension__cpp}
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__gpu}
    ${__CLDNN_Sources__cache}
    ${__CLDNN_Sources__ch_kernels}
    ${__CLDNN_Sources__cg_cache}
    ${__CLDNN_Sources__test_cases}
    ${__CLDNN_Sources__test_utils}
    ${__CLDNN_Sources__gtest}
  )
# Helping with some generators.
set_property(SOURCE ${__CLDNN_Sources__cg_cache} PROPERTY GENERATED TRUE)


# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__api}"                  FILES ${__CLDNN_Headers__api})
source_group("${__CLDNN_Label__api__cpp}"             FILES ${__CLDNN_Headers__api__cpp})
source_group("${__CLDNN_Label__api__c}"               FILES ${__CLDNN_Headers__api__c})
source_group("${__CLDNN_Label__api_extension}"        FILES ${__CLDNN_Headers__api_extension})
source_group("${__CLDNN_Label__api_extension__cpp}"   FILES ${__CLDNN_Headers__api_extension__cpp})
source_group("${__CLDNN_Label__api_extension__c}"     FILES ${__CLDNN_Headers__api_extension__c})
source_group("${__CLDNN_Label__include}"              FILES ${__CLDNN_Headers__include})
source_group("${__CLDNN_Label__graph_opt}"            FILES ${__CLDNN_Sources__graph_opt})
source_group("${__CLDNN_Label__caps}"                 FILES ${__CLDNN_Sources__caps})
source_group("${__CLDNN_Label__main}"                 FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__gpu}"                  FILES ${__CLDNN_Sources__gpu})
source_group("${__CLDNN_Label__cache}"                FILES ${__CLDNN_Sources__cache})
source_group("${__CLDNN_Label__ch_kernels}"           FILES ${__CLDNN_Sources__ch_kernels})
source_group("${__CLDNN_Label__cg_cache}"             FILES ${__CLDNN_Sources__cg_cache})
source_group("${__CLDNN_Label__test_cases}"           FILES ${__CLDNN_Sources__test_cases})
source_group("${__CLDNN_Label__test_utils}"           FILES ${__CLDNN_Sources__test_utils})
source_group("${__CLDNN_Label__gtest}"                FILES ${__CLDNN_Sources__gtest})


# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CLDNN__MAIN_DIR}/src"
    "${CLDNN__GTEST_DIR}"
    "${__CLDNN_Directory__test_utils}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${__CLDNN_Directory__include}"
    "${__CLDNN_IncDirectory__caps}"
    "${__CLDNN_Directory__ks_core}"
    "${__CLDNN_Directory__ks_core}/common"
    "${__CLDNN_Directory__ks_actual_kernels}"
    "${__CLDNN_Directory__ks_common}"
    "${__CLDNN_Directory__gpu}"
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
    # "${CLDNN_BUILD__PROJ__clDNN}"
    OpenCL
    cldnn_kernel_selector
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
