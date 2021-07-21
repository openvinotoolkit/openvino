# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# =================================== Name / Output settings from parent ===============================

set(CLDNN_BUILD__PROJ             "cldnn_runtime")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")

# ========================================= Source/Header files ========================================

set(__CLDNN_Directory__main           "${CMAKE_CURRENT_SOURCE_DIR}")
set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${__CLDNN_Directory__main}/*.h"
    "${__CLDNN_Directory__main}/*.hpp"
    "${__CLDNN_Directory__main}/*.cpp"
  )

file(GLOB __CLDNN_Sources__ocl
    "${__CLDNN_Directory__main}/ocl/*.h"
    "${__CLDNN_Directory__main}/ocl/*.hpp"
    "${__CLDNN_Directory__main}/ocl/*.cpp"
)

file(GLOB __CLDNN_Sources__ze
    "${__CLDNN_Directory__main}/ze/*.h"
    "${__CLDNN_Directory__main}/ze/*.hpp"
    "${__CLDNN_Directory__main}/ze/*.cpp"
)

set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__ocl}
)

if (GPU_BACKEND STREQUAL "L0")
  list(APPEND __CLDNN_AllSources ${__CLDNN_Sources__ze})
endif()

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"             FILES ${__CLDNN_Sources__main})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${__CLDNN_Directory__main}"
    $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
  )

# =================================== Link targets and dependencies ====================================

# Main static library.
add_library("${CLDNN_BUILD__PROJ}" STATIC
    ${__CLDNN_AllSources}
  )

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${CLDNN_BUILD__PROJ} PRIVATE -Os)
endif()
if (GPU_BACKEND STREQUAL "L0")
  target_compile_definitions("${CLDNN_BUILD__PROJ}" PUBLIC GPU_ENABLE_ZE_BACKEND)
endif()

set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")

if(COMMAND add_cpplint_target)
  add_cpplint_target("${CLDNN_BUILD__PROJ}_cpplint" FOR_TARGETS "${CLDNN_BUILD__PROJ}")
endif()

if(COMMAND set_ie_threading_interface_for)
  set_ie_threading_interface_for("${CLDNN_BUILD__PROJ}")
endif()

target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE
    OpenCL
    openvino::itt
    ${LevelZero_LIBRARY}
  )

if(ENABLE_ONEDNN_FOR_GPU)
  target_link_libraries("${CLDNN_BUILD__PROJ}" PUBLIC onednn_gpu)
endif()

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE pthread)
endif()

target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE inference_engine)

ov_install_static_lib("${CLDNN_BUILD__PROJ}" gpu)

# ======================================================================================================
