# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET gna_trans_lib_config)

set(SOURCES
    transformer_configuration.hpp
    transformer_configuration_loader.hpp
    transformer_configuration_loader_impl.hpp
    transformer_configuration_loader_impl.cpp)

add_library(${TARGET} STATIC ${SOURCES})

target_link_libraries(${TARGET} 
    PRIVATE openvino::runtime
    PUBLIC nlohmann_json::nlohmann_json)

add_clang_format_target(${TARGET}_clang FOR_TARGETS ${TARGET})
