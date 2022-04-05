# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumper)

list(APPEND LIBRARIES
        gflags
        inference_engine
        funcTestUtils
        pugixml::static
)

list(APPEND DEFINITIONS)

if(TARGET openvino_ir_frontend)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
    list(APPEND DEFINITIONS ENABLE_OV_IR_FRONTEND)
endif()

if(TARGET openvino_onnx_frontend)
    list(APPEND DEPENDENCIES openvino_onnx_frontend)
    list(APPEND DEFINITIONS ENABLE_OV_ONNX_FRONTEND)
endif()

if(TARGET openvino_paddle_frontend)
    list(APPEND DEPENDENCIES openvino_paddle_frontend)
    list(APPEND DEFINITIONS ENABLE_OV_PADDLE_FRONTEND)
endif()

if(TARGET openvino_tensorflow_frontend)
    list(APPEND DEPENDENCIES openvino_tensorflow_frontend)
    list(APPEND DEFINITIONS ENABLE_OV_TF_FRONTEND)
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        LINK_LIBRARIES
            PRIVATE
                ${LIBRARIES}
        DEPENDENCIES
            ${DEPENDENCIES}
        ADD_CPPLINT
)

target_compile_definitions(${TARGET_NAME} PRIVATE ${DEFINITIONS})
ie_faster_build(${TARGET_NAME} UNITY)
