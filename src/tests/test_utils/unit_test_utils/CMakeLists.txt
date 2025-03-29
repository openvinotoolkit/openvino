# Copyright (C) 2018-2025 Intel Corporation
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
        LINK_LIBRARIES
            PUBLIC
                common_test_utils_s
                openvino_runtime_s
                gmock
        DEPENDENCIES
            mock_engine
        INCLUDES
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
)

# install & export

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../unit_test_utils"
                                    INSTALL_DESTIONATION "developer_package/include/unit_test_utils/unit_test_utils")
