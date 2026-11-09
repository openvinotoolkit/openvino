# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME ir
    FILEDESCRIPTION "FrontEnd to load OpenVINO IR file format"
    LINK_LIBRARIES 
        openvino::pugixml
        openvino::core::dev
        openvino::xml_util
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/input_model.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/input_model.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.hpp
    PUBLIC_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/openvino/frontend/ir/frontend.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/openvino/frontend/ir/visibility.hpp)

ov_build_target_faster(openvino_ir_frontend PCH)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
