# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME subgraphsDumper)

list(APPEND LIBRARIES
        gflags
        openvino::runtime
        func_test_utils
        openvino::pugixml
)

list(APPEND DEPENDENCIES)

if(TARGET openvino_ir_frontend)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
endif()

if(TARGET openvino_onnx_frontend)
    list(APPEND DEPENDENCIES openvino_onnx_frontend)
endif()

if(TARGET openvino_paddle_frontend)
    list(APPEND DEPENDENCIES openvino_paddle_frontend)
endif()

if(TARGET openvino_pytorch_frontend)
    list(APPEND DEPENDENCIES openvino_pytorch_frontend)
endif()

if(TARGET openvino_tensorflow_frontend)
    list(APPEND DEPENDENCIES openvino_tensorflow_frontend)
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