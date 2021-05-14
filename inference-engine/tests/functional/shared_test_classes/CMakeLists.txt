# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME sharedTestClasses)

list(APPEND EXPORT_DEPENDENCIES
        funcTestUtils
        ngraphFunctions
        )

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            inference_engine_tests
        INCLUDES
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            PRIVATE
                ${EXPORT_DEPENDENCIES}
        EXPORT_DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
)

# CVS-55373
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/subgraph/softsign.cpp"
    PROPERTIES INCLUDE_DIRECTORIES
        $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)

ie_faster_build(${TARGET_NAME}
        PCH PRIVATE "src/precomp.hpp"
        )
