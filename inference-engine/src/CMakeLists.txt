# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
####################################
## All next project will use C++11
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
####################################

add_subdirectory(inference_engine)

if(ENABLE_MKL_DNN)
    add_subdirectory(mkldnn_plugin)
endif()

if(ENABLE_CLDNN)
    add_subdirectory(cldnn_engine)
endif()

add_subdirectory(hetero_plugin)

set(InferenceEngine_LIBRARIES inference_engine)
set(InferenceEngine_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)

# Fake set_target_cpu_flags() function. Currently it is used only for sample build.
# When IE will provide architecture optimized binary for each platform it will be switched on.
function(set_target_cpu_flags TARGET_NAME)
endfunction()

add_subdirectory(extension)
