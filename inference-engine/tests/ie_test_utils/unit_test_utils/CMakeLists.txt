# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME unitTestUtils)

add_subdirectory(mocks/mock_engine)

list(APPEND EXPORT_DEPENDENCIES
        commonTestUtils_s
        inference_engine_s
        gmock)

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            inference_engine_tests
        INCLUDES
            PRIVATE
                $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
        EXPORT_DEPENDENCIES
            ${EXPORT_DEPENDENCIES}
)

target_link_libraries(${TARGET_NAME}
        PUBLIC
        ${EXPORT_DEPENDENCIES})
