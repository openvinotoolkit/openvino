# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumper)

list(APPEND LIBRARIES
        gflags
        inference_engine
        funcTestUtils
        openvino::pugixml
)

if(ENABLE_OV_IR_FRONTEND)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
endif()

if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEPENDENCIES openvino_onnx_frontend)
endif()

if(ENABLE_OV_PADDLE_FRONTEND)
    list(APPEND DEPENDENCIES openvino_paddle_frontend)
endif()

# todo: use convert_model and save to ir instead this
if(ENABLE_OV_PYTORCH_FRONTEND)
    list(APPEND DEPENDENCIES openvino_pytorch_frontend)
endif()

if(ENABLE_OV_TF_FRONTEND)
    list(APPEND DEPENDENCIES openvino_tensorflow_frontend)
endif()

if(ENABLE_OV_TF_LITE_FRONTEND)
    list(APPEND DEPENDENCIES openvino_tensorflow_lite_frontend)
endif()

if(ENABLE_OV_PADDLE_FRONTEND)
    list(APPEND DEPENDENCIES openvino_paddle_frontend)
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src
        INCLUDES
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        LINK_LIBRARIES
            PRIVATE
                ${LIBRARIES}
        DEPENDENCIES
            ${DEPENDENCIES}
        ADD_CPPLINT
)

ie_faster_build(${TARGET_NAME} UNITY)