# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# =================================== Name / Output settings from parent ===============================

set(CLDNN_BUILD__PROJ             "cldnn_kernel_selector")
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

set(__CLDNN_Directory__core           "${__CLDNN_Directory__main}/core")
set(__CLDNN_Label__core               "core")
file(GLOB __CLDNN_Sources__core
    "${__CLDNN_Directory__core}/*.h"
    "${__CLDNN_Directory__core}/*.hpp"
    "${__CLDNN_Directory__core}/*.cpp"
  )

set(__CLDNN_Directory__common          "${__CLDNN_Directory__main}/common")
set(__CLDNN_Label__common              "common")
file(GLOB __CLDNN_Sources__common
    "${__CLDNN_Directory__common}/*.h"
    "${__CLDNN_Directory__common}/*.hpp"
    "${__CLDNN_Directory__common}/*.cpp"
  )

set(__CLDNN_Directory__core_common          "${__CLDNN_Directory__core}/common")
set(__CLDNN_Label__core_common              "${__CLDNN_Label__core}\\common")
file(GLOB __CLDNN_Sources__core_common
    "${__CLDNN_Directory__core_common}/*.h"
    "${__CLDNN_Directory__core_common}/*.hpp"
    "${__CLDNN_Directory__core_common}/*.cpp"
  )

set(__CLDNN_Directory__core_cache          "${__CLDNN_Directory__core}/cache")
set(__CLDNN_Label__core_cache              "${__CLDNN_Label__core}\\cache")
file(GLOB __CLDNN_Sources__core_cache
    "${__CLDNN_Directory__core_cache}/*.h"
    "${__CLDNN_Directory__core_cache}/*.hpp"
    "${__CLDNN_Directory__core_cache}/*.cpp"
  )

set(__CLDNN_Directory__actual_kernels "${__CLDNN_Directory__core}/actual_kernels")
set(__CLDNN_Label__actual_kernels     "${__CLDNN_Label__core}\\actual_kernels")
file(GLOB_RECURSE __CLDNN_Sources__actual_kernels
    "${__CLDNN_Directory__actual_kernels}/*.h"
    "${__CLDNN_Directory__actual_kernels}/*.hpp"
    "${__CLDNN_Directory__actual_kernels}/*.cpp"
  )
foreach(__CLDNN_FilePath ${__CLDNN_Sources__actual_kernels})
  string(REPLACE ";" "\;" __CLDNN_FilePath "${__CLDNN_FilePath}") # [WA#1] Must escape ; again if occurred in item.
  get_filename_component(__CLDNN_FileDir "${__CLDNN_FilePath}" DIRECTORY)
  get_filename_component(__CLDNN_DirName "${__CLDNN_FileDir}" NAME)

  set(__CLDNN_FileLabel "${__CLDNN_Label__actual_kernels}\\${__CLDNN_DirName}")
  source_group("${__CLDNN_FileLabel}" FILES ${__CLDNN_FilePath})
endforeach()

set(__CLDNN_Directory__cl_kernels      "${__CLDNN_Directory__core}/cl_kernels")
set(__CLDNN_Label__cl_kernels          "${__CLDNN_Label__core}\\cl_kernels")
file(GLOB_RECURSE __CLDNN_Sources__cl_kernels
    "${__CLDNN_Directory__cl_kernels}/*.cl"
  )

set(__CLDNN_Directory__cg_cache        "${CLDNN__CODEGEN_INCDIR}")
set(__CLDNN_CGDirectory__cg_cache      "${CLDNN__CODEGEN_DIR}/cache")
set(__CLDNN_Label__cg_cache            "${__CLDNN_Label__core}\\codegen")
set(__CLDNN_File__cg_cache__prim_db    "ks_primitive_db.inc")
set(__CLDNN_Sources__cg_cache
    "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
  )


set(__CLDNN_AllSources
    ${__CLDNN_Sources__main}
    ${__CLDNN_Sources__core}
    ${__CLDNN_Sources__common}
    ${__CLDNN_Sources__core_common}
    ${__CLDNN_Sources__core_cache}
    ${__CLDNN_Sources__actual_kernels}
    ${__CLDNN_Sources__cl_kernels}
    ${__CLDNN_Sources__cg_cache}
  )
# Helping with some generators.
set_property(SOURCE ${__CLDNN_Sources__cg_cache} PROPERTY GENERATED TRUE)

# =============================================== Filters ==============================================

source_group("${__CLDNN_Label__main}"             FILES ${__CLDNN_Sources__main})
source_group("${__CLDNN_Label__core}"             FILES ${__CLDNN_Sources__core})
source_group("${__CLDNN_Label__common}"           FILES ${__CLDNN_Sources__common})
source_group("${__CLDNN_Label__core_common}"      FILES ${__CLDNN_Sources__core_common})
source_group("${__CLDNN_Label__core_cache}"      FILES ${__CLDNN_Sources__core_cache})
#source_group("${__CLDNN_Label__actual_kernels}"  FILES ${__CLDNN_Sources__actual_kernels})
source_group("${__CLDNN_Label__cl_kernels}"       FILES ${__CLDNN_Sources__cl_kernels})
source_group("${__CLDNN_Label__cg_cache}"         FILES ${__CLDNN_Sources__cg_cache})

# ===================================== Include/Link directories =======================================

include_directories(
    "${CLDNN__MAIN_DIR}"
    "${__CLDNN_Directory__main}"
    "${__CLDNN_Directory__core}"
    "${__CLDNN_Directory__core}/common"
    "${__CLDNN_Directory__core}/cache"
    "${__CLDNN_Directory__actual_kernels}"
    "${__CLDNN_Directory__common}"
  )

# =================================== Link targets and dependencies ====================================

# Main shared library.
add_library("${CLDNN_BUILD__PROJ}" STATIC
    ${__CLDNN_AllSources}
  )

set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY PROJECT_LABEL "${CLDNN_BUILD__PROJ_LABEL}")
set_property(TARGET "${CLDNN_BUILD__PROJ}" PROPERTY OUTPUT_NAME   "${CLDNN_BUILD__PROJ_OUTPUT_NAME}")

if(COMMAND add_cpplint_target)
  add_cpplint_target("${CLDNN_BUILD__PROJ}_cpplint" FOR_TARGETS "${CLDNN_BUILD__PROJ}")
endif()

target_link_libraries("${CLDNN_BUILD__PROJ}"
    OpenCL
  )

if(WIN32)
  target_link_libraries("${CLDNN_BUILD__PROJ}" setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries("${CLDNN_BUILD__PROJ}" pthread)
endif()
target_link_libraries("${CLDNN_BUILD__PROJ}" ${CLDNN__SYSTEM_LINK_LIBRARIES})

# =================================== Custom pre- and post-steps =======================================
add_custom_command(OUTPUT "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${__CLDNN_CGDirectory__cg_cache}"
    COMMAND "${PYTHON_EXECUTABLE}" "${__CLDNN_Directory__core_common}/primitive_db_gen.py" -out_path "${__CLDNN_CGDirectory__cg_cache}" -out_file_name "${__CLDNN_File__cg_cache__prim_db}" -kernels "${__CLDNN_Directory__cl_kernels}"
    DEPENDS ${__CLDNN_Sources__cl_kernels} "${__CLDNN_Directory__core_common}/primitive_db_gen.py"
    COMMENT "Generating ${__CLDNN_File__cg_cache__prim_db} ..."
  )
add_custom_command(OUTPUT "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}" "${__CLDNN_Directory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}"
    DEPENDS "${__CLDNN_CGDirectory__cg_cache}/${__CLDNN_File__cg_cache__prim_db}" ${__CLDNN_Sources__cl_kernels} "${__CLDNN_Directory__core_common}/primitive_db_gen.py"
    COMMENT "Updating file if the file changed (${__CLDNN_File__cg_cache__prim_db}) ..."
  )
if(WIN32)
  set(CLDNN_CACHE_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>")
else((NOT ANDROID) AND (UNIX))
  set(CLDNN_CACHE_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/")
endif()

add_custom_command(
    TARGET "${CLDNN_BUILD__PROJ}" POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${__CLDNN_Directory__core}/cache/cache.json ${CLDNN_CACHE_PATH}/cache.json)


# ======================================================================================================
