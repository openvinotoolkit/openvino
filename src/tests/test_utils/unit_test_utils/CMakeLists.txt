# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

set(TARGET_NAME unit_test_utils)

add_subdirectory(mocks/mock_engine)

ov_add_target(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADD_CLANG_FORMAT
        DEVELOPER_PACKAGE
            tests
        LINK_LIBRARIES
            PUBLIC
                common_test_utils_s
                inference_engine_s
                gmock
        DEPENDENCIES
            mock_engine
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
                "$<INSTALL_INTERFACE:developer_package/include/unit_test_utils>"
)

# install

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION developer_package/include/unit_test_utils/unit_test_utils
        COMPONENT developer_package EXCLUDE_FROM_ALL)
