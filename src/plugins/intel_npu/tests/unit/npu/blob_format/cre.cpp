// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cre.hpp"

std::vector<CREParams> valid_test_cases = {
    std::make_tuple("expr_1_none", expression_1, std::vector<uint16_t>{}, true),
    // shouldn't this return true?
    std::make_tuple("expr1_none", expression_2, std::vector<uint16_t>{}, false),

    std::make_tuple("expr1__ELF", expression_3, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, true),
    std::make_tuple("expr1__none", expression_3, std::vector<uint16_t>{}, false),

    std::make_tuple("expr2__ELF_BS", expression_4, std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING}, true),
    std::make_tuple("expr2__BS", expression_4, std::vector<uint16_t>{CRE::BATCHING}, false),
    std::make_tuple("expr2__ELF", expression_4, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, false),

    std::make_tuple("expr3__ELF_BS_WS",
                    expression_5,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr3__ELF_WS",
                    expression_5,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION},
                    false),
    std::make_tuple("expr3__ELF", expression_5, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, false),

    std::make_tuple("expr4_ELF_BS_WS",
                    expression_6,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr4_ELF_BS", expression_6, std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING}, true),
    std::make_tuple("expr4_ELF_WS",
                    expression_6,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr4_BS_WS", expression_6, std::vector<uint16_t>{CRE::BATCHING, CRE::WEIGHTS_SEPARATION}, false),
    std::make_tuple("expr4_none", expression_6, std::vector<uint16_t>{}, false),

    std::make_tuple("expr5_ELF_BS_WS",
                    expression_7,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr5_ELF", expression_7, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, true),
    std::make_tuple("expr5_BS_WS", expression_7, std::vector<uint16_t>{CRE::BATCHING, CRE::WEIGHTS_SEPARATION}, true),
    std::make_tuple("expr5_WS", expression_7, std::vector<uint16_t>{CRE::WEIGHTS_SEPARATION}, false),

    std::make_tuple("expr6_ELF_BS_WS",
                    expression_8,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr6_ELF_BS", expression_8, std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING}, true),
    std::make_tuple("expr6_ELF_WS",
                    expression_8,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr6_BS_WS", expression_8, std::vector<uint16_t>{CRE::BATCHING, CRE::WEIGHTS_SEPARATION}, false),

    std::make_tuple("expr_7_ELF_BS_WS",
                    expression_9,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr_7_WS", expression_9, std::vector<uint16_t>{CRE::WEIGHTS_SEPARATION}, true),
    std::make_tuple("expr_7_ELF", expression_9, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, true),
    std::make_tuple("expr_7_BS", expression_9, std::vector<uint16_t>{CRE::BATCHING}, false),
    std::make_tuple("expr_7_none", expression_9, std::vector<uint16_t>{}, false),

    std::make_tuple("expr_8_ELF_BS_WS",
                    expression_10,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr_8_WS", expression_10, std::vector<uint16_t>{CRE::WEIGHTS_SEPARATION}, true),
    std::make_tuple("expr_8_none", expression_10, std::vector<uint16_t>{}, false),

    std::make_tuple("expr_9_ELF_BS_WS",
                    expression_11,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr_9_ELF_WS",
                    expression_11,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr_9_ELF_BS", expression_11, std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING}, true),
    std::make_tuple("expr_9_BS_WS",
                    expression_11,
                    std::vector<uint16_t>{CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    false),
    std::make_tuple("expr_9_ELF", expression_11, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, false),
};

INSTANTIATE_TEST_SUITE_P(CRETests,
                         ValidExpression,
                         ::testing::ValuesIn(valid_test_cases),
                         CREUnitTests::getTestCaseName);

std::vector<CREParams> invalid_test_cases = {
    std::make_tuple("invalid_expr_1", invalid_expression_1, std::vector<uint16_t>{}, false),
    std::make_tuple("invalid_expr_2", invalid_expression_2, std::vector<uint16_t>{}, false),
    std::make_tuple("invalid_expr_3", invalid_expression_3, std::vector<uint16_t>{}, false),
    std::make_tuple("invalid_expr_4", invalid_expression_4, std::vector<uint16_t>{}, false),
    std::make_tuple("invalid_expr_5", invalid_expression_5, std::vector<uint16_t>{}, false),
};

INSTANTIATE_TEST_SUITE_P(CRETests,
                         InvalidExpression,
                         ::testing::ValuesIn(invalid_test_cases),
                         CREUnitTests::getTestCaseName);
