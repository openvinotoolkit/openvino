// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cre.hpp"

#define MAKE_PARAM(expr, result, ...) \
    std::make_tuple(std::string(#expr), expr, std::vector<uint16_t>{__VA_ARGS__}, result)

std::vector<CREParams> valid_test_cases = {
    MAKE_PARAM(expression_1, true),
    // shouldn't this return true?
    MAKE_PARAM(expression_2, false),

    MAKE_PARAM(expression_3, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_3, false),

    MAKE_PARAM(expression_4, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_4, false, CRE::BATCHING),
    MAKE_PARAM(expression_4, false, CRE::ELF_SCHEDULE),

    MAKE_PARAM(expression_5, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_5, false, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_5, false, CRE::ELF_SCHEDULE),

    MAKE_PARAM(expression_6, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_6, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_6, true, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_6, false, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_6, false),

    MAKE_PARAM(expression_7, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_7, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_7, true, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_7, false, CRE::WEIGHTS_SEPARATION),

    MAKE_PARAM(expression_8, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_8, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_8, true, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_8, false, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),

    MAKE_PARAM(expression_9, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_9, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_9, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_9, false, CRE::BATCHING),
    MAKE_PARAM(expression_9, false),

    // should have the same behavior as expression_9
    MAKE_PARAM(expression_10, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_10, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_10, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_10, false, CRE::BATCHING),
    MAKE_PARAM(expression_10, false),

    MAKE_PARAM(expression_11, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_11, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_11, false),

    MAKE_PARAM(expression_12, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_12, true, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_12, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_12, false, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_12, false, CRE::ELF_SCHEDULE),

    MAKE_PARAM(expression_13, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
};

INSTANTIATE_TEST_SUITE_P(CRETests,
                         ValidExpression,
                         ::testing::ValuesIn(valid_test_cases),
                         CREUnitTests::getTestCaseName);

std::vector<CREParams> invalid_test_cases = {
    MAKE_PARAM(invalid_expression_1, false),
    MAKE_PARAM(invalid_expression_2, false),
    MAKE_PARAM(invalid_expression_3, false),
    MAKE_PARAM(invalid_expression_4, false),
    MAKE_PARAM(invalid_expression_5, false),
};

INSTANTIATE_TEST_SUITE_P(CRETests,
                         InvalidExpression,
                         ::testing::ValuesIn(invalid_test_cases),
                         CREUnitTests::getTestCaseName);
