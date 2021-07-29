# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

set(TARGET_NAME unitTestUtils)

add_subdirectory(mocks/mock_engine)

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            inference_engine_tests
        LINK_LIBRARIES
            PUBLIC
                commonTestUtils_s
                inference_engine_s
                gmock
        INCLUDES
            PRIVATE
                $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>
)
