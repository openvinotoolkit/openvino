# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME sharedTestClasses)

addIeTarget(
        NAME ${TARGET_NAME}
        TYPE STATIC
        ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            tests
        INCLUDES
            PUBLIC
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
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
