# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Flags for 3rd party projects
#

set(use_static_runtime ON)

if(use_static_runtime)
    foreach(lang C CXX)
        foreach(build_type "" "_DEBUG" "_MINSIZEREL" "_RELEASE" "_RELWITHDEBINFO")
            set(flag_var "CMAKE_${lang}_FLAGS${build_type}")
            string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endforeach()
    endforeach()
endif()

function(onecoreuap_set_runtime var)
    set(${var} ${use_static_runtime} CACHE BOOL "" FORCE)
endfunction()

# ONNX
onecoreuap_set_runtime(ONNX_USE_MSVC_STATIC_RUNTIME)
# pugixml
onecoreuap_set_runtime(STATIC_CRT)
# protobuf
onecoreuap_set_runtime(protobuf_MSVC_STATIC_RUNTIME)
# clDNN
onecoreuap_set_runtime(CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME)
# google-test
if(use_static_runtime)
    set(gtest_force_shared_crt OFF CACHE BOOL "" FORCE)
else()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

unset(use_static_runtime)
