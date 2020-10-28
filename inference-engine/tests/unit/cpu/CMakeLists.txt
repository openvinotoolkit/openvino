# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME cpuUnitTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${IE_MAIN_SOURCE_DIR}/src/mkldnn_plugin
            ${IE_MAIN_SOURCE_DIR}/src/transformations/include
        OBJECT_FILES
            $<TARGET_OBJECTS:MKLDNNPlugin_obj>
        LINK_LIBRARIES
            unitTestUtils
            mkldnn
            inference_engine_transformations
        ADD_CPPLINT
        LABELS
            CPU
)

if(USE_CNNNETWORK_LPT)
    target_link_libraries(${TARGET_NAME} PRIVATE inference_engine_lp_transformations_legacy)
else()
    target_link_libraries(${TARGET_NAME} PRIVATE inference_engine_lp_transformations)
endif()

ie_faster_build(${TARGET_NAME}
    UNITY
)
