# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME funcSharedTests)

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(DEPENDENCIES inference_engine mock_engine HeteroPlugin MultiDevicePlugin)
if (NGRAPH_ONNX_FRONTEND_ENABLE)
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
            inference_engine_tests
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
                inference_engine_transformations
        DEPENDENCIES
            ${DEPENDENCIES}
)

# CVS-55376
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/low_precision_transformations/layer_transformation.cpp"
    PROPERTIES INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)

ie_faster_build(${TARGET_NAME}
    PCH PRIVATE "src/precomp.hpp"
)

if (TARGET MKLDNNPlugin)
    add_dependencies(${TARGET_NAME} MKLDNNPlugin)
endif()
