// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cre.hpp"

std::vector<CREParams> test_cases = {
    std::make_tuple("expr1__ELF", expression_1, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, true),
    std::make_tuple("expr1__none", expression_1, std::vector<uint16_t>{}, false),

    std::make_tuple("expr2__ELF_BS", expression_2, std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING}, true),
    std::make_tuple("expr2__BS", expression_2, std::vector<uint16_t>{CRE::BATCHING}, false),
    std::make_tuple("expr2__ELF", expression_2, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, false),

    std::make_tuple("expr3__ELF_BS_WS",
                    expression_3,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr3__ELF_WS",
                    expression_3,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION},
                    false),
    std::make_tuple("expr3__ELF", expression_3, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, false),

    std::make_tuple("expr4_ELF_BS_WS",
                    expression_4,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr4_ELF_BS", expression_4, std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING}, true),
    std::make_tuple("expr4_ELF_WS",
                    expression_4,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr4_BS_WS", expression_4, std::vector<uint16_t>{CRE::BATCHING, CRE::WEIGHTS_SEPARATION}, false),
    std::make_tuple("expr4_none", expression_4, std::vector<uint16_t>{}, false),

    std::make_tuple("expr5_ELF_BS_WS",
                    expression_5,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr5_ELF", expression_5, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, true),
    std::make_tuple("expr5_BS_WS", expression_5, std::vector<uint16_t>{CRE::BATCHING, CRE::WEIGHTS_SEPARATION}, true),
    std::make_tuple("expr5_WS", expression_5, std::vector<uint16_t>{CRE::WEIGHTS_SEPARATION}, false),

    std::make_tuple("expr6_ELF_BS_WS",
                    expression_6,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr6_ELF_BS", expression_6, std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING}, true),
    std::make_tuple("expr6_ELF_WS",
                    expression_6,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr6_BS_WS", expression_6, std::vector<uint16_t>{CRE::BATCHING, CRE::WEIGHTS_SEPARATION}, false),

    std::make_tuple("expr_7_ELF_BS_WS",
                    expression_7,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr_7_WS", expression_7, std::vector<uint16_t>{CRE::WEIGHTS_SEPARATION}, true),
    std::make_tuple("expr_7_ELF", expression_7, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, true),
    std::make_tuple("expr_7_BS", expression_7, std::vector<uint16_t>{CRE::BATCHING}, false),
    std::make_tuple("expr_7_none", expression_7, std::vector<uint16_t>{}, false),

    std::make_tuple("expr_8_ELF_BS_WS",
                    expression_8,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr_8_WS", expression_8, std::vector<uint16_t>{CRE::WEIGHTS_SEPARATION}, true),
    std::make_tuple("expr_8_none", expression_8, std::vector<uint16_t>{}, false),

    std::make_tuple("expr_9_ELF_BS_WS",
                    expression_9,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr_9_ELF_WS",
                    expression_9,
                    std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION},
                    true),
    std::make_tuple("expr_9_ELF_BS", expression_9, std::vector<uint16_t>{CRE::ELF_SCHEDULE, CRE::BATCHING}, true),
    std::make_tuple("expr_9_BS_WS", expression_9, std::vector<uint16_t>{CRE::BATCHING, CRE::WEIGHTS_SEPARATION}, false),
    std::make_tuple("expr_9_ELF", expression_9, std::vector<uint16_t>{CRE::ELF_SCHEDULE}, false),
};

INSTANTIATE_TEST_SUITE_P(CRETests, CREUnitTests, ::testing::ValuesIn(test_cases), CREUnitTests::getTestCaseName);
