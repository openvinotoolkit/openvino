# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
        OBJECT_FILES
            $<TARGET_OBJECTS:MKLDNNPlugin_obj>
        LINK_LIBRARIES
            gtest
            gtest_main
            mkldnn
            inference_engine_transformations
            inference_engine_lp_transformations
            inference_engine_s
            inference_engine_snippets
            ngraphFunctions
        ADD_CPPLINT
        LABELS
            CPU
)

ie_faster_build(${TARGET_NAME}
    UNITY
)

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/common
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/src/cpu
        ${IE_MAIN_SOURCE_DIR}/thirdparty/mkl-dnn/include)
