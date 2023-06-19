# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET gna_transformer_app_passes)

SET(HEADERS
    app_pass_manager.hpp
    insert_add_after_parameter.hpp)

SET(SOURCES
    app_pass_manager.cpp
    insert_add_after_parameter.cpp)

add_library(${TARGET} STATIC ${HEADERS} ${SOURCES})
target_link_libraries(${TARGET} PUBLIC openvino::core::dev openvino::conditional_compilation)
target_include_directories(${TARGET} PUBLIC .)

add_clang_format_target(${TARGET}_clang FOR_TARGETS ${TARGET})
