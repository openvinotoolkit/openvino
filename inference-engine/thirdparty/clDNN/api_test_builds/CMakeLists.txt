# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# =================================== Name / Output settings from parent ===============================

set(CLDNN_BUILD__PROJ             "api_test_builds")
set(CLDNN_BUILD__PROJ_LABEL       "API test builds")
set(CLDNN_BUILD__PROJ_OUTPUT_DIR  "${CLDNN__CODEGEN_BASEDIR}/out")
set(CLDNN_BUILD__PROJ_OUTPUT_NAME "api_test_builds${CLDNN__OUT_CPU_SUFFIX}")

# ================================== Compiler preprocessor definitions =================================

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    CLDNN_EXPORTS
    EXPORT_NEURAL_SYMBOLS
  )

# =========================================== Compiler options =========================================

intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET
      StandardCxx11
  )
intel_config_flag_apply_settings(CompilerOptions CMAKE_C_FLAGS ALL_PATTERN ""
    SET
      StandardC99
  )

# ========================================= Source/Header files ========================================

set(__CLDNN_Label__api                  "api")
file(GLOB __CLDNN_Headers__api
    "${CLDNN__API_DIR}/*.hpp"
  )

set(__CLDNN_Directory__api__cpp         "${CLDNN__API_DIR}/CPP")
set(__CLDNN_Label__api__cpp             "${__CLDNN_Label__api}\\CPP")
file(GLOB __CLDNN_Headers__api__cpp
    "${__CLDNN_Directory__api__cpp}/*.hpp"
  )

set(__CLDNN_Directory__test_builds      "${CLDNN__CODEGEN_DIR}/test_builds")
set(__CLDNN_CGDirectory__test_builds    "${CLDNN__CODEGEN_BASEDIR}/test_builds")
set(__CLDNN_Label__test_builds          "test builds\\codegen")
set(__CLDNN_File__test_builds__api__cpp "api_cpp_test.cpp")
set(__CLDNN_Sources__test_builds
    "${__CLDNN_Directory__test_builds}/${__CLDNN_File__test_builds__api__cpp}"
  )

set(__CLDNN_AllSources
    ${__CLDNN_Headers__api}
    ${__CLDNN_Headers__api__cpp}
    ${__CLDNN_Headers__api__c}
    ${__CLDNN_Sources__test_builds}
  )

# Helping with some generators.
set_property(SOURCE ${__CLDNN_Sources__test_builds} PROPERTY GENERATED TRUE)

# ======================================= Pre-generating test files ====================================

file(MAKE_DIRECTORY "${__CLDNN_CGDirectory__test_builds}")

# C++ API testing (C++11 compatibility).
set(__CLDNN_CGFile__api__cpp "${__CLDNN_CGDirectory__test_builds}/${__CLDNN_File__test_builds__api__cpp}")
file(WRITE "${__CLDNN_CGFile__api__cpp}" "// This file is auto-generated. Please, do not modify it directly.\n\n")
foreach(__CLDNN_Header ${__CLDNN_Headers__api} ${__CLDNN_Headers__api__cpp})
  string(REPLACE ";" "\;" __CLDNN_Header "${__CLDNN_Header}") # [WA#1] Must escape ; again if occurred in item.
  file(APPEND "${__CLDNN_CGFile__api__cpp}" "#include \"${__CLDNN_Header}\"\n")
endforeach()
file(APPEND "${__CLDNN_CGFile__api__cpp}" "\n")

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__api}"         FILES ${__CLDNN_Headers__api})
source_group("${__CLDNN_Label__api__cpp}"    FILES ${__CLDNN_Headers__api__cpp})
source_group("${__CLDNN_Label__test_builds}" FILES ${__CLDNN_Sources__test_builds})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
  )

# =================================== Link targets and dependencies ====================================

# Test static library.
add_library("${CLDNN_BUILD__PROJ}" STATIC
    ${__CLDNN_AllSources}
  )
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL            "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY ARCHIVE_OUTPUT_DIRECTORY "${CLDNN_BUILD__PROJ_OUTPUT_DIR}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME              "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")

target_link_libraries("${CLDNN_BUILD__PROJ}" ${CLDNN__SYSTEM_LINK_LIBRARIES})


add_dependencies("${CLDNN_BUILD__PROJ__clDNN}"
    "${CLDNN_BUILD__PROJ}"
  )

# =================================== Custom pre- and post-steps =======================================

add_custom_command(OUTPUT "${__CLDNN_Directory__test_builds}/${__CLDNN_File__test_builds__api__cpp}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${__CLDNN_CGFile__api__cpp}" "${__CLDNN_Directory__test_builds}/${__CLDNN_File__test_builds__api__cpp}"
    DEPENDS "${__CLDNN_CGFile__api__cpp}" ${__CLDNN_Headers__api} ${__CLDNN_Headers__api__cpp}
    COMMENT "Updating file if the file changed (${__CLDNN_File__test_builds__api__cpp}) ..."
  )

# ======================================================================================================
