# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

SET(GNA_TRANS_TRANSFORMER gna_trans_lib_transformer_test_app)
add_executable(${GNA_TRANS_TRANSFORMER}
    model_transformer_app_test.cpp
    model_transformer_gna_test.cpp
)

target_link_libraries(${GNA_TRANS_TRANSFORMER} gtest gtest_main gna_trans_lib_transformer inference_engine_s openvino::conditional_compilation)

add_test(
    NAME ${GNA_TRANS_TRANSFORMER}
    COMMAND ${GNA_TRANS_TRANSFORMER}
)

add_clang_format_target(${GNA_TRANS_TRANSFORMER}_clang FOR_TARGETS ${GNA_TRANS_TRANSFORMER})