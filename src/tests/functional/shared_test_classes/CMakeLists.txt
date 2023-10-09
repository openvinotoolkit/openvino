# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME sharedTestClasses)

ov_add_target(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            tests
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                "$<INSTALL_INTERFACE:developer_package/include/sharedTestClasses>"
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        LINK_LIBRARIES
            PRIVATE
                func_test_utils
                ov_models
)

ov_build_target_faster(${TARGET_NAME}
        PCH PRIVATE "src/precomp.hpp"
        )

# install

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION developer_package/include/sharedTestClasses
        COMPONENT developer_package EXCLUDE_FROM_ALL)
