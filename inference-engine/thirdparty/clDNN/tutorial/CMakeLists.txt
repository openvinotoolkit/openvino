# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ========================================= Name / Output settings =====================================

set(CLDNN_BUILD__PROJ             "tutorial")
set(CLDNN_BUILD__PROJ_LABEL       "${CLDNN_BUILD__PROJ}")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "${CLDNN_BUILD__PROJ}${CLDNN__OUT_CPU_SUFFIX}")

# =========================================== Compiler options =========================================

intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET
      StandardCxx11
      RttiEnabled
      WarnLevel3
      TreatWarnAsErrorDisabled
  )

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__main                "")
file(GLOB __CLDNN_Sources__main
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )


set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
  )

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"   FILES ${__CLDNN_Sources__main})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )

# =================================== Link targets and dependencies ====================================

# Examples executable.
add_executable("${CLDNN_BUILD__PROJ}"
    ${__CLDNN_AllSources}
  )
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")

target_link_libraries("${CLDNN_BUILD__PROJ}"
    "${CLDNN_BUILD__PROJ__clDNN}"
  )
target_link_libraries("${CLDNN_BUILD__PROJ}" ${CLDNN__SYSTEM_LINK_LIBRARIES} OpenCL)

# ======================================================================================================
