# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# =================================== Name / Output settings from parent ===============================

set(CLDNN_BUILD__PROJ             "${CLDNN_BUILD__PROJ__clDNN}")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ_LABEL__clDNN}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ_OUTPUT_NAME__clDNN}")

# ========================================== Compiler options ==========================================
# FIXME: latest findings showed issue with using libgomp inside clDNN
#        temporarily disabling OpenMP to ensure clDNN works
#find_package(OpenMP)
if (OPENMP_FOUND)
    add_definitions(-DOPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

intel_config_flag_apply_settings(
    CompilerOptions
    CMAKE_CXX_FLAGS
    ALL_PATTERN ""
    SET
      RttiEnabled
  )

# ================================== Compiler preprocessor definitions =================================

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    "CLDNN_VERSION_MAJOR=${CLDNN__VERSION_MAJOR}"
    "CLDNN_VERSION_MINOR=${CLDNN__VERSION_MINOR}"
    "CLDNN_VERSION_BUILD=${CLDNN__VERSION_BUILD}"
    "CLDNN_VERSION_REVISION=${CLDNN__VERSION_REVISION}"
  )

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__api                 "api")
file(GLOB_RECURSE __CLDNN_Headers__api
    "${CLDNN__API_DIR}/*.h"
    "${CLDNN__API_DIR}/*.hpp"
  )

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

set(__CLDNN_Directory__graph_opt       "${CMAKE_CURRENT_SOURCE_DIR}/graph_optimizer")
set(__CLDNN_Label__graph_opt           "graph_optimizer")
file(GLOB __CLDNN_Sources__graph_opt
    "${__CLDNN_Directory__graph_opt}/*.h"
    "${__CLDNN_Directory__graph_opt}/*.hpp"
    "${__CLDNN_Directory__graph_opt}/*.cpp"
  )

set(__CLDNN_Directory__include         "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(__CLDNN_Label__include             "include")
file(GLOB __CLDNN_Headers__include
    "${__CLDNN_Directory__include}/*.h"
    "${__CLDNN_Directory__include}/*.hpp"
  )

set(__CLDNN_Directory__gpu             "${CMAKE_CURRENT_SOURCE_DIR}/gpu")
set(__CLDNN_Label__gpu                 "gpu")
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
    ${__CLDNN_Headers__api}
    ${__CLDNN_Sources__graph_opt}
    ${__CLDNN_Headers__include}
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__gpu}
    ${__CLDNN_Sources__cache}
    ${__CLDNN_Sources__ch_kernels}
    ${__CLDNN_Sources__cg_cache}
  )
# Helping with some generators.
set_property(SOURCE ${__CLDNN_Sources__cg_cache} PROPERTY GENERATED TRUE)

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__api}"                  FILES ${__CLDNN_Headers__api})
source_group("${__CLDNN_Label__include}"              FILES ${__CLDNN_Headers__include})
source_group("${__CLDNN_Label__graph_opt}"            FILES ${__CLDNN_Sources__graph_opt})
source_group("${__CLDNN_Label__main}"                 FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__gpu}"                  FILES ${__CLDNN_Sources__gpu})
source_group("${__CLDNN_Label__cache}"                FILES ${__CLDNN_Sources__cache})
source_group("${__CLDNN_Label__ch_kernels}"           FILES ${__CLDNN_Sources__ch_kernels})
source_group("${__CLDNN_Label__cg_cache}"             FILES ${__CLDNN_Sources__cg_cache})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CLDNN__RUNTIME_DIR}"
    "${__CLDNN_Directory__include}"
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

target_link_libraries("${CLDNN_BUILD__PROJ}" PUBLIC
    OpenCL
  )

target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE
    cldnn_kernel_selector
    openvino::itt
    cldnn_runtime
  )

if(COMMAND add_cpplint_target)
  add_cpplint_target("${CLDNN_BUILD__PROJ}_cpplint" FOR_TARGETS "${CLDNN_BUILD__PROJ}")
endif()

if(COMMAND set_ie_threading_interface_for)
  set_ie_threading_interface_for("${CLDNN_BUILD__PROJ}")
endif()

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE pthread)
endif()
target_link_libraries("${CLDNN_BUILD__PROJ}" PRIVATE ${CLDNN__SYSTEM_LINK_LIBRARIES})

# ======================================================================================================
