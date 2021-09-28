# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (SRC
    all_close_f.cpp
    engine/ie_engines.cpp
    engine/interpreter_engine.cpp
    engine/shared_utils.cpp
    test_common.cpp
    test_common.hpp
    comparator_test_utils.cpp
    comparator_test_utils.hpp
    float_util.cpp
    test_tools.cpp
    test_case.cpp
    test_control.cpp
    visitor.hpp
    provenance_enabler.hpp
)
if (NGRAPH_ONNX_FRONTEND_ENABLE)
    list(APPEND SRC onnx_test_util.cpp)
endif()

add_library(ngraph_test_util STATIC ${SRC})

if(COMMAND ie_faster_build)
    ie_faster_build(ngraph_test_util
        UNITY
    )
endif()

target_include_directories(ngraph_test_util PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
    $<TARGET_PROPERTY:inference_engine,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(ngraph_test_util PUBLIC ngraph ngraph_backend gtest gmock PRIVATE frontend_manager::static)
if (NGRAPH_ONNX_FRONTEND_ENABLE)
    target_link_libraries(ngraph_test_util PRIVATE onnx_common)
endif()
target_include_directories(ngraph_test_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

openvino_developer_export_targets(COMPONENT ngraph TARGETS ngraph_test_util)
