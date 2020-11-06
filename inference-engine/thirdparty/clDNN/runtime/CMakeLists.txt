# Copyright (c) 2020 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =================================== Name / Output settings from parent ===============================

set(CLDNN_BUILD__PROJ             "cldnn_runtime")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# ========================================== Compiler options ==========================================

intel_config_flag_apply_settings(
    CompilerOptions
    CMAKE_CXX_FLAGS
    ALL_PATTERN ""
    SET
      RttiEnabled
  )

# ================================== Compiler preprocessor definitions =================================

if (CLDNN__ENABLE_PERSISTENT_CACHE)
  set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS CLDNN_ENABLE_PERSISTENT_CACHE)
endif ()

# ========================================= Source/Header files ========================================

set(__CLDNN_Directory__main           "${CMAKE_CURRENT_SOURCE_DIR}")
set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${__CLDNN_Directory__main}/*.h"
    "${__CLDNN_Directory__main}/*.hpp"
    "${__CLDNN_Directory__main}/*.cpp"
  )

set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
  )

set(__CLDNN_Directory__ks_main           "${CLDNN__KERNEL_SELECTOR_DIR}")
set(__CLDNN_Directory__ks_core           "${CLDNN__KERNEL_SELECTOR_DIR}/core")
set(__CLDNN_Directory__ks_common         "${CLDNN__KERNEL_SELECTOR_DIR}/common")
set(__CLDNN_Directory__ks_core_common    "${__CLDNN_Directory__ks_core}/common")
set(__CLDNN_Directory__ks_actual_kernels "${__CLDNN_Directory__ks_core}/actual_kernels")
set(__CLDNN_Directory__ks_cache          "${__CLDNN_Directory__ks_core}/cache")

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"             FILES ${__CLDNN_Sources__main})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${__CLDNN_Directory__main}"
    "${__CLDNN_Directory__ks_core}"
    "${__CLDNN_Directory__ks_core}/common"
    "${__CLDNN_Directory__ks_actual_kernels}"
    "${__CLDNN_Directory__ks_common}"
  )

# =================================== Link targets and dependencies ====================================

# Main static library.
add_library("${CLDNN_BUILD__PROJ}" STATIC
    ${__CLDNN_AllSources}
  )

set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")


target_link_libraries("${CLDNN_BUILD__PROJ}"
    clDNN_OpenCL
  )

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries("${CLDNN_BUILD__PROJ}" pthread)
endif()
target_link_libraries("${CLDNN_BUILD__PROJ}" ${CLDNN__SYSTEM_LINK_LIBRARIES})

# ======================================================================================================
