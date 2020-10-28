# Copyright (C) 2016-2020 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME IESharedTests)

disable_deprecated_warnings()

list(APPEND SHARED_LIBRARIES
        ${NGRAPH_LIBRARIES}
        ie_tests
        ngraphFunctions
        )

if(USE_CNNNETWORK_LPT)
    file(GLOB SHARED_TESTS_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/common_single_layer_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lstm/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_tools/*.cpp
        # requires legacy LPT
        ${CMAKE_CURRENT_SOURCE_DIR}/network_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transformations/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transformations/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/transformations/common/*.cpp
        )
    list(APPEND SHARED_LIBRARIES inference_engine_lp_transformations_legacy)
else()
    file(GLOB SHARED_TESTS_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/common_single_layer_tests/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lstm/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_tools/*.cpp
        )
endif()

add_library(${TARGET_NAME} STATIC ${SHARED_TESTS_SRC})
add_dependencies(${TARGET_NAME} inference_engine_preproc MultiDevicePlugin mock_engine)

ie_faster_build(${TARGET_NAME}
    UNITY
    PCH PRIVATE "precomp.hpp"
)

if(ENABLE_MKL_DNN)
    add_dependencies(${TARGET_NAME} MKLDNNPlugin)
    target_compile_definitions(${TARGET_NAME} PUBLIC ENABLE_MKL_DNN)
endif()

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs imgproc QUIET)
if(OpenCV_FOUND)
    target_compile_definitions(${TARGET_NAME} PUBLIC USE_OPENCV)
else()
    message(WARNING "No suitable OpenCV version detected, pre-processing tests are skipped in " ${TARGET_NAME})
endif()

target_include_directories(${TARGET_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/network_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/io_blob_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/input_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/inference_engine_regression_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/lstm
        ${CMAKE_CURRENT_SOURCE_DIR}/common_single_layer_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_tools
        ${CMAKE_CURRENT_SOURCE_DIR}/transformations
        $<TARGET_PROPERTY:inference_engine_plugin_api,INTERFACE_INCLUDE_DIRECTORIES>
)

if(OpenCV_FOUND)
    target_include_directories(${TARGET_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    list(APPEND SHARED_LIBRARIES ${OpenCV_LIBS})
endif()

target_link_libraries(${TARGET_NAME} PUBLIC ${SHARED_LIBRARIES})

add_dependencies(${TARGET_NAME} HeteroPlugin)

# developer package

ie_developer_export_targets(${TARGET_NAME})
