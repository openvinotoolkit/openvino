# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            $<TARGET_PROPERTY:MKLDNNPlugin,SOURCE_DIR>
            $<TARGET_PROPERTY:MKLDNNPlugin,SOURCE_DIR>/nodes
            $<TARGET_PROPERTY:openvino::conditional_compilation,INTERFACE_INCLUDE_DIRECTORIES>
        OBJECT_FILES
            $<TARGET_OBJECTS:MKLDNNPlugin_obj>
        LINK_LIBRARIES
            gtest
            gtest_main
            gmock
            mkldnn
            inference_engine_transformations
            inference_engine_lp_transformations
            ov_shape_inference
            inference_engine_s
            unitTestUtils
        ADD_CPPLINT
        LABELS
            CPU
)

ie_faster_build(${TARGET_NAME}
    UNITY
)
