# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

SET(GNA_TRANS_CONFIG gna_trans_lib_config_test_app)
add_executable(${GNA_TRANS_CONFIG}
    transformer_configuration_loader_impl_test.cpp
)

target_link_libraries(${GNA_TRANS_CONFIG} gtest gtest_main gna_trans_lib_config)

add_test(
    NAME ${GNA_TRANS_CONFIG}
    COMMAND ${GNA_TRANS_CONFIG}
)

add_clang_format_target(${GNA_TRANS_CONFIG}_clang FOR_TARGETS ${GNA_TRANS_CONFIG})