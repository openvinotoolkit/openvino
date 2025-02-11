# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# =================================================================
#
# Generates cpp file with dispatcher for cross compiled function
# Parameters:
#   XARCH_API_HEADER -- path to header with function declaration
#   XARCH_FUNC_NAMES -- names of functions to dispatch
#   XARCH_NAMESPACES -- full namespace used to keep ODR
#   XARCH_DISP_FILE -- dispatcher file name to generate
#   XARCH_SET -- set of ARCH supported by dispatcher. semicolon-delimited
#
# =================================================================

set(_CPU_CHECK_ANY         "true")
set(_CPU_CHECK_SSE42       "with_cpu_x86_sse42()")
set(_CPU_CHECK_AVX         "with_cpu_x86_avx()")
set(_CPU_CHECK_NEON_FP16   "with_cpu_neon_fp16()")
set(_CPU_CHECK_SVE         "with_cpu_sve()")
set(_CPU_CHECK_AVX2        "with_cpu_x86_avx2()")
set(_CPU_CHECK_AVX512F     "with_cpu_x86_avx512f()")

function(_generate_dispatcher)
    string(REPLACE "::" ";" XARCH_NAMESPACES "${XARCH_NAMESPACES}")

    list(GET XARCH_NAMESPACES -1 XARCH_CURRENT_NAMESPACE)
    set(PARENT_NAMESPACES ${XARCH_NAMESPACES})
    list(REMOVE_AT PARENT_NAMESPACES -1)

    set(DISP_CONTENT
"
//
// Auto generated file by CMake macros cross_compiled_file()
// !! do not modify it !!!
//
#include \"${XARCH_API_HEADER}\"
#include \"openvino/runtime/system_conf.hpp\"

")

    foreach(_namespace IN LISTS PARENT_NAMESPACES)
        string(APPEND DISP_CONTENT
            "namespace ${_namespace} {\n")
    endforeach()

    foreach(_func_name IN LISTS XARCH_FUNC_NAMES)
        _find_signature_in_file(${XARCH_API_HEADER} ${_func_name} SIGNATURE)
        _generate_call_line_from_signature("${SIGNATURE}" CALL_LINE)

        foreach(_arch IN LISTS XARCH_SET)
            string(APPEND DISP_CONTENT
                "namespace ${_arch} {\n    ${SIGNATURE}\; \n}\n")
        endforeach()

        string(APPEND DISP_CONTENT
                "namespace ${XARCH_CURRENT_NAMESPACE} {\n\n${SIGNATURE} {\n")

        foreach(_arch IN LISTS XARCH_SET)
            string(APPEND DISP_CONTENT
                "    if (${_CPU_CHECK_${_arch}}) {\n        return ${_arch}::${CALL_LINE}\;\n    }\n")
        endforeach()

        string(APPEND DISP_CONTENT "}\n\n}\n")
    endforeach()

    foreach(_namespace IN LISTS PARENT_NAMESPACES)
        string(APPEND DISP_CONTENT "}  // namespace ${_namespace}\n")
    endforeach()

    file(WRITE ${XARCH_DISP_FILE} ${DISP_CONTENT})
endfunction()


function(_find_signature_in_file FILE FUNCTION RESULT_NAME)
    file(READ "${FILE}" CONTENT)
    set(valid_chars "<>:_*& a-zA-Z0-9\n") ## valid chars for type/var specification (including new line /n)
    string(REGEX MATCH "[${valid_chars}]*${FUNCTION}[ ]*[(][=,${valid_chars}]*[)]" SIGNATURE ${CONTENT})
    string(STRIP "${SIGNATURE}" SIGNATURE)
    set (${RESULT_NAME} "${SIGNATURE}" PARENT_SCOPE)
endfunction()

function(_generate_call_line_from_signature SIGNATURE RESULT_NAME)
    ## extract func name
    set(_name ${SIGNATURE})
    string(REGEX REPLACE "[ ]*[(].*[)]" "" _name "${_name}")   # remove arguments
    string(REGEX MATCH "[a-zA-Z0-9_]*[ ]*$" _name "${_name}")      # extract func name

    set(nt_chars "[:_*& a-zA-Z0-9\n]*") ## any sequence of chars to describe object type (no template)

    ## extract arg names
    set(_args ${SIGNATURE})
    string(REGEX MATCH "[(].*[)]"  _args "${_args}")   # extract args with types, all inside brackets
    string(REGEX REPLACE "<${nt_chars},${nt_chars}>" "" _args "${_args}") # remove template brackets with ','
    string(REPLACE "(" "" _args ${_args})
    string(REPLACE ")" "" _args ${_args})
    string(REPLACE "," ";" _args ${_args})   # now it's list
    foreach(_arg_elem ${_args})
        string(REGEX MATCH "[a-zA-Z0-9_]*[ ]*$" _arg_elem "${_arg_elem}")
        list(APPEND _arg_names ${_arg_elem})
    endforeach()
    string(REPLACE ";" ", " _arg_names "${_arg_names}")  # back to comma separated string

    set (${RESULT_NAME} "${_name}(${_arg_names})" PARENT_SCOPE)
endfunction()

_generate_dispatcher()
