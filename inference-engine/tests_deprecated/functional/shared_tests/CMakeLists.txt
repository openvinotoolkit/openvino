# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME IESharedTests)

list(APPEND SHARED_LIBRARIES
        ${NGRAPH_LIBRARIES}
        commonTestUtils
        ngraphFunctions
        ieTestHelpers
        )

file(GLOB SHARED_TESTS_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/common_single_layer_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lstm/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_tools/*.cpp
        )

add_library(${TARGET_NAME} STATIC ${SHARED_TESTS_SRC})
add_dependencies(${TARGET_NAME} inference_engine_preproc MultiDevicePlugin mock_engine)

set_ie_threading_interface_for(${TARGET_NAME})

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "precomp.hpp"
)

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS core imgproc QUIET)
if(OpenCV_FOUND)
    target_compile_definitions(${TARGET_NAME} PUBLIC USE_OPENCV)
else()
    message(WARNING "No suitable OpenCV version detected, pre-processing tests are skipped in ${TARGET_NAME}")
endif()

target_include_directories(${TARGET_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/io_blob_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/input_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/lstm
        ${CMAKE_CURRENT_SOURCE_DIR}/common_single_layer_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
)

if(OpenCV_FOUND)
    target_include_directories(${TARGET_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    list(APPEND SHARED_LIBRARIES ${OpenCV_LIBS})
endif()

target_link_libraries(${TARGET_NAME} PUBLIC ${SHARED_LIBRARIES})

add_dependencies(${TARGET_NAME} HeteroPlugin)

# developer package

openvino_developer_export_targets(COMPONENT inference_engine_tests TARGETS ${TARGET_NAME})
