# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (SRC
    all_close_f.cpp
    engine/ie_engines.cpp
    engine/interpreter_engine.cpp
    engine/shared_utils.cpp
    float_util.cpp
    test_tools.cpp
    test_case.cpp
    test_control.cpp
    visitor.hpp
    provenance_enabler.hpp
)
if (NGRAPH_ONNX_IMPORT_ENABLE)
    list(APPEND SRC onnx_test_util.cpp)
endif()

add_library(ngraph_test_util STATIC ${SRC})

if(COMMAND ie_faster_build)
    ie_faster_build(ngraph_test_util
        UNITY
    )
endif()

target_include_directories(ngraph_test_util PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${IE_MAIN_SOURCE_DIR}/include)
target_link_libraries(ngraph_test_util PUBLIC ngraph ngraph_backend gtest gmock)
if (NGRAPH_ONNX_IMPORT_ENABLE)
    target_link_libraries(ngraph_test_util PRIVATE onnx_common)
endif()
