# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME funcSharedTests)

list(APPEND EXPORT_DEPENDENCIES
        funcTestUtils
        ngraphFunctions
        lptNgraphFunctions
        sharedTestClasses
        )

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${PUBLIC_HEADERS_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            inference_engine_tests
        INCLUDES
            PUBLIC
                ${PUBLIC_HEADERS_DIR}
        LINK_LIBRARIES
            PUBLIC
                ${EXPORT_DEPENDENCIES}
            PRIVATE
                inference_engine_transformations
        DEPENDENCIES
            inference_engine
            mock_engine
            HeteroPlugin
            MultiDevicePlugin
        EXPORT_DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
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
