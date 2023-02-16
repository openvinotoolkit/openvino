# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gna_tests_commons)

#SET(HEADERS tests_commons/library.hpp)
#SET(SOURCES tests_commons/library.cpp)

#add_library(${TARGET_NAME} STATIC ${HEADERS} ${SOURCES})
#target_link_libraries(${TARGET_NAME} ngraphFunctions openvino_intel_gna_plugin_test_static gtest)

#target_include_directories(${TARGET_NAME}  PUBLIC ${CMAKE_CURRENT_LIST_DIR})


addIeTarget(
        NAME ${TARGET_NAME}_unit
        TYPE STATIC
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/tests_commons
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            tests
        INCLUDES
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            PRIVATE
                gtest
                ngraphFunctions
                openvino_intel_gna_plugin_test_static
)

addIeTarget(
        NAME ${TARGET_NAME}_func
        TYPE STATIC
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/tests_commons
        ADD_CPPLINT
        DEVELOPER_PACKAGE
            tests
        INCLUDES
            PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}
            PRIVATE
                ${OpenVINO_SOURCE_DIR}/src/plugins/intel_gna/src/
        LINK_LIBRARIES
            PRIVATE
                 gtest
                 ngraphFunctions
        DEPENDENCIES
            openvino_intel_gna_plugin
)
