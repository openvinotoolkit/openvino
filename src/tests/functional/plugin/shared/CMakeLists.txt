# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME funcSharedTests)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(DEPENDENCIES mock_engine)

if(ENABLE_HETERO)
    list(APPEND DEPENDENCIES openvino_hetero_plugin)
endif()

if(ENABLE_AUTO OR ENABLE_MULTI)
    list(APPEND DEPENDENCIES openvino_auto_plugin)
endif()

if(ENABLE_AUTO_BATCH)
    list(APPEND DEPENDENCIES openvino_auto_batch_plugin)
endif()


# remove once CVS-69781 is fixed
if(ENABLE_OV_IR_FRONTEND)
    list(APPEND DEPENDENCIES openvino_ir_frontend)
endif()

if (ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEPENDENCIES test_model_zoo)
    list(APPEND DEFINES TEST_MODELS="${TEST_MODEL_ZOO}/func_tests/models/")
else()
    set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx)
endif()

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            tests
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
        DEFINES ${DEFINES}
        INCLUDES
            PUBLIC
                ${PUBLIC_HEADERS_DIR}
        LINK_LIBRARIES
            PUBLIC
                pugixml::static
                funcTestUtils
                ngraph_test_util
                ngraphFunctions
                lptNgraphFunctions
                sharedTestClasses
            PRIVATE
                openvino::util
                openvino::runtime::dev
                snippetsNgraphFunctions
        DEPENDENCIES
            ${DEPENDENCIES}
)

if(ENABLE_GAPI_PREPROCESSING)
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_GAPI_PREPROCESSING)
endif()

ie_faster_build(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)
