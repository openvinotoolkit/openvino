# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET gna_trans_lib_transformer)

set(SOURCES
    model_transformer.hpp
    model_transformer_gna.hpp
    model_transformer_gna.cpp
    model_transformer_app.hpp
    model_transformer_app.cpp
)

add_library(${TARGET} STATIC ${SOURCES})

target_link_libraries(${TARGET}
    PRIVATE openvino_intel_gna_plugin_test_static inference_engine_legacy_s
    PUBLIC gna_trans_lib_config gna_trans_lib_logger libGNA inference_engine_s)

add_clang_format_target(${TARGET}_clang FOR_TARGETS ${TARGET})