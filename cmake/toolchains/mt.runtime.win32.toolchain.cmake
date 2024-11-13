# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Flags for 3rd party projects
#

set(use_static_runtime ON)

if(use_static_runtime)
    set(use_dynamic_runtime OFF)
else()
    set(use_dynamic_runtime ON)
endif()

# CMAKE_MSVC_RUNTIME_LIBRARY is available since cmake 3.15
if(use_static_runtime AND CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "^MultiThreaded.*DLL$")
    message(FATAL_ERROR "Misleading configuration, CMAKE_MSVC_RUNTIME_LIBRARY is ${CMAKE_MSVC_RUNTIME_LIBRARY}")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<BOOL:${use_dynamic_runtime}>:DLL>)
endif()

if(use_static_runtime)
    foreach(lang C CXX)
        foreach(build_type "" "_DEBUG" "_MINSIZEREL" "_RELEASE" "_RELWITHDEBINFO")
            set(flag_var "CMAKE_${lang}_FLAGS${build_type}_INIT")
            string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            if (build_type STREQUAL "_DEBUG")
                set(${flag_var} "/MTd")
            else()
                set(${flag_var} "/MT")
            endif()
        endforeach()
    endforeach()
endif()

macro(ov_set_msvc_runtime var value)
    if(NOT DEFINED ${var})
        set(${var} ${value} CACHE BOOL "" FORCE)
    endif()
endmacro()

# static TBBBind_2_5 is built with dynamic CRT runtime
ov_set_msvc_runtime(ENABLE_TBBBIND_2_5 ${use_dynamic_runtime})
# ONNX
ov_set_msvc_runtime(ONNX_USE_MSVC_STATIC_RUNTIME ${use_static_runtime})
ov_set_msvc_runtime(ONNX_USE_MSVC_SHARED_RUNTIME ${use_dynamic_runtime})
# pugixml
ov_set_msvc_runtime(STATIC_CRT ${use_static_runtime})
# protobuf
ov_set_msvc_runtime(protobuf_MSVC_STATIC_RUNTIME ${use_static_runtime})
# clDNN
ov_set_msvc_runtime(CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME ${use_static_runtime})
# OpenCL
ov_set_msvc_runtime(USE_DYNAMIC_VCXX_RUNTIME ${use_dynamic_runtime})
# google-test
ov_set_msvc_runtime(gtest_force_shared_crt ${use_dynamic_runtime})

unset(use_static_runtime)
unset(use_dynamic_runtime)
