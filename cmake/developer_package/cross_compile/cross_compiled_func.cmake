# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

## list of available instruction sets
set(_AVAILABLE_ARCHS_LIST ANY SSE42 AVX AVX2 AVX512F NEON_FP16 SVE)

if(ENABLE_SVE)
    list(APPEND _ENABLED_ARCHS_LIST SVE)
endif()
if(ENABLE_NEON_FP16)
    list(APPEND _ENABLED_ARCHS_LIST NEON_FP16)
endif()
if(ENABLE_AVX512F)
    list(APPEND _ENABLED_ARCHS_LIST AVX512F)
endif()
if(ENABLE_AVX2)
    list(APPEND _ENABLED_ARCHS_LIST AVX2)
endif()
if(ENABLE_SSE42)
    list(APPEND _ENABLED_ARCHS_LIST SSE42)
endif()
list(APPEND _ENABLED_ARCHS_LIST ANY)

## Arch specific definitions
set(_DEFINE_ANY       "")
set(_DEFINE_SSE42     "HAVE_SSE42"    ${_DEFINE_ANY})
set(_DEFINE_AVX       "HAVE_AVX"      ${_DEFINE_SSE42})
set(_DEFINE_AVX2      "HAVE_AVX2"     ${_DEFINE_AVX})
set(_DEFINE_AVX512F   "HAVE_AVX512F"  ${_DEFINE_AVX2})
set(_DEFINE_NEON_FP16 "HAVE_NEON_FP16" ${_DEFINE_ANY})
set(_DEFINE_SVE       "HAVE_SVE"      ${_DEFINE_SVE})

## Arch specific compile options
if(ENABLE_AVX512F)
    ov_avx512_optimization_flags(_FLAGS_AVX512F)
endif()
if(ENABLE_AVX2)
    ov_avx2_optimization_flags(_FLAGS_AVX2)
endif()
if(ENABLE_SSE42)
    ov_sse42_optimization_flags(_FLAGS_SSE42)
endif()
if(ENABLE_NEON_FP16)
    ov_arm_neon_fp16_optimization_flags(_FLAGS_NEON_FP16)
endif()
if(ENABLE_SVE)
    ov_arm_sve_optimization_flags(_FLAGS_SVE)
endif()
set(_FLAGS_AVX "")  ## TBD is not defined for OV project yet
set(_FLAGS_ANY "")  ##

## way to duplicate file via cmake tool set
if (UNIX)
    ## Clone sources via sym link because it allow to modify original file in IDE along with debug
    set(TO_DUPLICATE create_symlink)
else()
    ## Windows and others - just copy
    set(TO_DUPLICATE copy)
endif()

set(DISPATCHER_GEN_SCRIPT         ${CMAKE_CURRENT_LIST_DIR}/cross_compiled_disp_gen.cmake)
set(DISPATCHER_GEN_OPTIONS_HOLDER ${CMAKE_CURRENT_LIST_DIR}/cross_compiled_disp_gen_options.in)


#######################################
#
#  Allow to enable multiple cross compilation of source file inside one module
#  with keeping requirements on minimal instruction set. The CPU check performed
#  in runtime via common utils declared in "system_conf.h".
#
#  Usage example:
#  cross_compiled_file(<target>
#         ARCH
#            ANY       <source_file>
#            SSE SSE42 <source_file>
#            AVX AVX2  <source_file>
#            AVX512F   <source_file>
#         API <header_file>
#         NAMESPACE <namespace>   # like "IE::Ext::CPU::XARCH"
#         NAME <function_names>    # like "my_fun1 my_fun2"
#     )
#
function(cross_compiled_file TARGET)
    set(oneValueArgs   API        ## Header with declaration of cross compiled function
                       NAMESPACE  ## The namespace where cross compiled function was declared
                       )
    set(multiValueArgs NAME       ## String with function signatures to make cross compiled
                       ARCH)      ## List of architecture described in _ARCH_LIST
    cmake_parse_arguments(X "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    ## verification
    if(X_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown argument: " ${X_UNPARSED_ARGUMENTS})
    endif()
    if((NOT TARGET) OR (NOT X_NAME) OR (NOT X_NAMESPACE) OR (NOT X_API) OR (NOT X_ARCH))
        message(FATAL_ERROR "Missed arguments in 'cross_compiled_file'")
    endif()

    ## format: ARCH1 ARCH2 <src1> ARCH3 <src2> ...
    foreach(_it IN LISTS X_ARCH)
        if(_it IN_LIST _AVAILABLE_ARCHS_LIST)
            ## that is arch ID
            set(_arch ${_it})
            if(_arch IN_LIST _ENABLED_ARCHS_LIST)
                # make non/less-optimized version coming first
                list(INSERT _CUR_ARCH_SET 0 ${_arch})
                list(APPEND _FULL_ARCH_SET ${_arch})
            endif()
        else()
            ## that is source file name
            set(_src_name ${_it})
            _remove_source_from_target(${TARGET} ${_src_name})

            if(_CUR_ARCH_SET)
                _clone_source_to_target(${TARGET} ${_src_name} "${_CUR_ARCH_SET}")
                unset(_CUR_ARCH_SET)
            endif()
        endif()
    endforeach()

    _add_dispatcher_to_target(${TARGET} ${X_API} "${X_NAME}" "${X_NAMESPACE}" "${_FULL_ARCH_SET}")
endfunction()


##########################################
#
#  Add source multiple time per each element in ARCH_SET.
#  Also provide corresponding arch specific flags and defines.
#
function(_clone_source_to_target TARGET SOURCE ARCH_SET)
    foreach(_arch ${ARCH_SET})
        set(_arch_dir cross-compiled/${_arch})

        get_filename_component(ARCH_NAME ${SOURCE} NAME)
        get_filename_component(ARCH_INCLUDE_DIR ${SOURCE} DIRECTORY)
        set(ARCH_SOURCE "${_arch_dir}/${ARCH_NAME}")

        add_custom_command(
                OUTPUT  ${ARCH_SOURCE}
                COMMAND ${CMAKE_COMMAND} -E make_directory
                        ${CMAKE_CURRENT_BINARY_DIR}/${_arch_dir}
                COMMAND ${CMAKE_COMMAND} -E ${TO_DUPLICATE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
                        ${CMAKE_CURRENT_BINARY_DIR}/${ARCH_SOURCE}
                DEPENDS ${SOURCE}
                VERBATIM
                )

        set_property(SOURCE ${ARCH_SOURCE} APPEND_STRING PROPERTY COMPILE_OPTIONS
                "${_FLAGS_${_arch}}")

        set_property(SOURCE ${ARCH_SOURCE} APPEND PROPERTY COMPILE_DEFINITIONS
                ${_DEFINE_${_arch}}
                "XARCH=${_arch}" ## to replace XARCH with direct ARCH name
                )

        ## To make `#include "some.hpp"` valid
        set_property(SOURCE ${ARCH_SOURCE} APPEND PROPERTY INCLUDE_DIRECTORIES
                "${CMAKE_CURRENT_SOURCE_DIR}/${ARCH_INCLUDE_DIR}")

        list(APPEND _ARCH_SOURCES ${ARCH_SOURCE})
    endforeach()

    target_sources(${TARGET} PRIVATE ${_ARCH_SOURCES})
endfunction()


##########################################
#
#  Generate dispatcher for provided function
#  for archs in ARCH_SET.
#
function(_add_dispatcher_to_target TARGET HEADER FUNC_NAME NAMESPACE ARCH_SET)
    get_filename_component(DISPATCHER_NAME ${HEADER} NAME_WE)
    get_filename_component(DISPATCHER_INCLUDE_DIR ${HEADER} DIRECTORY)
    set(DISPATCHER_SOURCE     "cross-compiled/${DISPATCHER_NAME}_disp.cpp")
    set(DISPATCHER_OPT_HOLDER "cross-compiled/${DISPATCHER_NAME}_holder.txt")

    configure_file(${DISPATCHER_GEN_OPTIONS_HOLDER} ${DISPATCHER_OPT_HOLDER})

    add_custom_command(
            OUTPUT  ${DISPATCHER_SOURCE}
            COMMAND ${CMAKE_COMMAND}
                    -D "XARCH_FUNC_NAMES=${X_NAME}"
                    -D "XARCH_NAMESPACES=${NAMESPACE}"
                    -D "XARCH_API_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/${HEADER}"
                    -D "XARCH_DISP_FILE=${CMAKE_CURRENT_BINARY_DIR}/${DISPATCHER_SOURCE}"
                    -D "XARCH_SET=${ARCH_SET}"
                    -P ${DISPATCHER_GEN_SCRIPT}
            DEPENDS ${HEADER}
                    ${DISPATCHER_GEN_SCRIPT}
                    ${CMAKE_CURRENT_BINARY_DIR}/${DISPATCHER_OPT_HOLDER} ## Just to make run dependency on args value
            VERBATIM
    )

    set_property(SOURCE ${DISPATCHER_SOURCE} APPEND PROPERTY INCLUDE_DIRECTORIES
            "${CMAKE_CURRENT_SOURCE_DIR}/${DISPATCHER_INCLUDE_DIR}")

    target_sources(${TARGET} PRIVATE ${DISPATCHER_SOURCE})
endfunction()

#####################################
#
#  Utils to handle with cmake target
#
function(_remove_source_from_target TARGET SOURCE_FILE)
    get_target_property(ORIGINAL_SOURCES ${TARGET} SOURCES)

    ## To match by file name only. The path is any.
    list(FILTER ORIGINAL_SOURCES EXCLUDE REGEX ".*${SOURCE_FILE}$")

    set_target_properties(${TARGET}
            PROPERTIES
            SOURCES "${ORIGINAL_SOURCES}")
endfunction()
