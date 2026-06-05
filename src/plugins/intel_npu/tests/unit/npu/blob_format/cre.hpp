// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "intel_npu/common/cre.hpp"
#include "intel_npu/common/supported_section_type_evaluator.hpp"
#include "mocks/mock_capabilities.hpp"

using namespace intel_npu;

using CREParams = std::tuple<std::string, std::vector<CRE::Token>, std::vector<uint16_t>, bool>;
// TODO CRE section tests
class CREUnitTests : public ::testing::TestWithParam<CREParams> {
protected:
    void SetUp() override {
        std::string name;
        std::vector<CRE::Token> expression;
        std::vector<uint16_t> supported_capabilities_tokens;
        std::tie(name, expression, supported_capabilities_tokens, is_compatible) = GetParam();

        cre = CRE(expression);

        for (const auto& token : supported_capabilities_tokens) {
            supported_capabilities[token] = std::make_shared<SupportedSectionTypeEvaluator>(token);
        }
    }

    CRE cre;
    std::unordered_map<CRE::Token, std::shared_ptr<ISectionTypeEvaluator>> supported_capabilities;
    bool is_compatible;

public:
    static std::string tokenToString(CRE::Token token) {
        switch (token) {
        case CRE::CRE_EVALUATION:
            return "CRE_EVAL";
        case CRE::ELF_SCHEDULE:
            return "ELF";
        case CRE::BATCHING:
            return "BATCH";
        case CRE::WEIGHTS_SEPARATION:
            return "WS";
        default:
            return std::to_string(token);
        }
    }

    static std::string getTestCaseName(testing::TestParamInfo<CREParams> obj) {
        const auto& [name, expression, section_types, is_compatible] = obj.param;

        std::string caps_str = section_types.empty() ? "none" : "";
        for (size_t i = 0; i < section_types.size(); i++) {
            if (i > 0) {
                caps_str += "_";
            }
            caps_str += tokenToString(section_types[i]);
        }

        return name + "__" + caps_str + "__" + (is_compatible ? "compatible" : "incompatible");
    }
};

using ValidExpression = CREUnitTests;

TEST_P(ValidExpression, check_compatibility) {
    EXPECT_EQ(cre.check_compatibility(supported_capabilities), is_compatible);
}

using InvalidExpression = CREUnitTests;

TEST_P(InvalidExpression, check_compatibility) {
    EXPECT_THROW(cre.check_compatibility(supported_capabilities), InvalidCRE);
}

using CREAppendSingleToken = ::testing::Test;

TEST_F(CREAppendSingleToken, AppendValidTokenUpdatesExpression) {
    CRE cre;
    cre.append_to_expression(CRE::ELF_SCHEDULE);
    EXPECT_EQ(cre.get_expression_length(), 1);
    EXPECT_EQ(cre.get_expression(), (std::vector<CRE::Token>{CRE::ELF_SCHEDULE}));
}

TEST_F(CREAppendSingleToken, AppendMultipleValidTokensAccumulates) {
    CRE cre;
    cre.append_to_expression(CRE::ELF_SCHEDULE);
    cre.append_to_expression(CRE::BATCHING);
    cre.append_to_expression(CRE::WEIGHTS_SEPARATION);
    EXPECT_EQ(cre.get_expression_length(), 5);
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CRE::Token>{CRE::ELF_SCHEDULE, CRE::AND, CRE::BATCHING, CRE::AND, CRE::WEIGHTS_SEPARATION}));
}

TEST_F(CREAppendSingleToken, AppendReservedTokenThrows) {
    CRE cre;
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::AND));
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::OR));
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::OPEN));
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::CLOSE));
    EXPECT_ANY_THROW(cre.append_to_expression(CRE::NOT));
}

TEST_F(CREAppendSingleToken, BuildsEvaluableAndExpression) {
    CRE cre;
    cre.append_to_expression(CRE::ELF_SCHEDULE);
    cre.append_to_expression(CRE::BATCHING);

    std::unordered_map<CRE::Token, std::shared_ptr<ISectionTypeEvaluator>> caps;
    caps[CRE::ELF_SCHEDULE] = std::make_shared<SupportedSectionTypeEvaluator>(CRE::ELF_SCHEDULE);
    caps[CRE::BATCHING] = std::make_shared<SupportedSectionTypeEvaluator>(CRE::BATCHING);
    EXPECT_TRUE(cre.check_compatibility(caps));

    caps.erase(CRE::BATCHING);
    EXPECT_FALSE(cre.check_compatibility(caps));
}

using CREAppendToken = ::testing::Test;

TEST_F(CREAppendToken, AppendEmptyVector) {
    CRE cre;
    cre.append_to_expression(std::vector<CRE::Token>{});
    EXPECT_EQ(cre.get_expression_length(), 0);
    EXPECT_EQ(cre.get_expression(), (std::vector<CRE::Token>{}));
}

TEST_F(CREAppendToken, AppendSubexpressionTokens) {
    CRE cre;
    cre.append_to_expression(
        std::vector<CRE::Token>{CRE::OPEN, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CRE::Token>{CRE::OPEN, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE}));

    cre.append_to_expression(std::vector<CRE::Token>{CRE::OPEN, CRE::BATCHING, CRE::CLOSE});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CRE::Token>{CRE::OPEN,
                                       CRE::BATCHING,
                                       CRE::OR,
                                       CRE::WEIGHTS_SEPARATION,
                                       CRE::CLOSE,
                                       CRE::AND,
                                       CRE::OPEN,
                                       CRE::BATCHING,
                                       CRE::CLOSE}));
}

/**
 * @brief Upon appending a subexpression, the CRE will add parrethesis automatically if the subexpression is longer than
 * two tokens (to include a valid binary operator) and the expression is not already enclosed.
 */
TEST_F(CREAppendToken, AppendSubexpressionAddsParrentheses) {
    CRE cre;
    cre.append_to_expression(std::vector<CRE::Token>{CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION});
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CRE::Token>{CRE::OPEN, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE}));

    cre = {};
    cre.append_to_expression(std::vector<CRE::Token>{CRE::OPEN, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION});
    EXPECT_EQ(
        cre.get_expression(),
        (std::vector<CRE::Token>{CRE::OPEN, CRE::OPEN, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE}));

    cre = {};
    cre.append_to_expression(std::vector<CRE::Token>{CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE});
    EXPECT_EQ(
        cre.get_expression(),
        (std::vector<CRE::Token>{CRE::OPEN, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE, CRE::CLOSE}));
}

/**
 * @brief Parretheses are not necessary if the subexpression has less than three tokens (no valid binary operator can be
 * there) or if the subexpression is already enclosed.
 */
TEST_F(CREAppendToken, AppendSubexpressionWithoutParrentheses) {
    CRE cre;
    cre.append_to_expression(std::vector<CRE::Token>{CRE::NOT, CRE::BATCHING});
    EXPECT_EQ(cre.get_expression(), (std::vector<CRE::Token>{CRE::NOT, CRE::BATCHING}));

    cre = {};
    cre.append_to_expression(std::vector<CRE::Token>{CRE::OPEN, CRE::NOT, CRE::BATCHING, CRE::CLOSE});
    EXPECT_EQ(cre.get_expression(), (std::vector<CRE::Token>{CRE::OPEN, CRE::NOT, CRE::BATCHING, CRE::CLOSE}));
}

/**
 * @brief The CRE code should be able to detect duplicate subexpressions (relative to depth level 0) and avoid inserting
 * copies.
 */
TEST_F(CREAppendToken, AvoidAppendingDuplicates) {
    CRE cre;
    cre.append_to_expression(CRE::BATCHING);
    cre.append_to_expression(CRE::BATCHING);
    EXPECT_EQ(cre.get_expression(), (std::vector<CRE::Token>{CRE::BATCHING}));

    cre = {};
    cre.append_to_expression(
        std::vector<CRE::Token>{CRE::OPEN, CRE::NOT, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE});
    cre.append_to_expression(
        std::vector<CRE::Token>{CRE::OPEN, CRE::NOT, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE});
    EXPECT_EQ(
        cre.get_expression(),
        (std::vector<CRE::Token>{CRE::OPEN, CRE::NOT, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE}));
}

TEST_F(CREAppendToken, MixedAppend) {
    CRE cre;
    cre.append_to_expression(CRE::ELF_SCHEDULE);
    cre.append_to_expression(
        std::vector<CRE::Token>{CRE::OPEN, CRE::BATCHING, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE});

    std::unordered_map<CRE::Token, std::shared_ptr<ISectionTypeEvaluator>> caps;
    caps[CRE::ELF_SCHEDULE] = std::make_shared<SupportedSectionTypeEvaluator>(CRE::ELF_SCHEDULE);
    caps[CRE::BATCHING] = std::make_shared<SupportedSectionTypeEvaluator>(CRE::BATCHING);
    EXPECT_TRUE(cre.check_compatibility(caps));

    caps.erase(CRE::ELF_SCHEDULE);
    EXPECT_FALSE(cre.check_compatibility(caps));
}

class CREOperandsEvaluation : public ::testing::Test {
protected:
    void SetUp() override {
        cap_1 = std::make_shared<MockCapability>(MockTypes::MOCK_1);
        cap_2 = std::make_shared<MockCapability>(MockTypes::MOCK_2);
        cap_3 = std::make_shared<MockCapability>(MockTypes::MOCK_3);

        caps[MockTypes::MOCK_1] = cap_1;
        caps[MockTypes::MOCK_2] = cap_2;
        caps[MockTypes::MOCK_3] = cap_3;
    }

    std::shared_ptr<MockCapability> cap_1;
    std::shared_ptr<MockCapability> cap_2;
    std::shared_ptr<MockCapability> cap_3;
    std::unordered_map<CRE::Token, std::shared_ptr<ISectionTypeEvaluator>> caps;
};

TEST_F(CREOperandsEvaluation, Depth0ORs) {
    EXPECT_CALL(*cap_1, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, lazy_check_support()).Times(0);

    CRE cre({MockTypes::MOCK_1, CRE::OR, MockTypes::MOCK_2, CRE::OR, MockTypes::MOCK_2});

    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, Depth0ANDs) {
    EXPECT_CALL(*cap_1, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, lazy_check_support()).Times(0);

    CRE cre({CRE::NOT, MockTypes::MOCK_1, CRE::AND, MockTypes::MOCK_2, CRE::AND, MockTypes::MOCK_2});

    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, Depth0AllEvaluate) {
    EXPECT_CALL(*cap_1, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_3, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));

    CRE cre({CRE::NOT, MockTypes::MOCK_1, CRE::OR, MockTypes::MOCK_2, CRE::AND, MockTypes::MOCK_3});

    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, ORFollowedByAND) {
    EXPECT_CALL(*cap_1, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, lazy_check_support()).Times(0);

    CRE cre(
        {CRE::NOT, CRE::OPEN, MockTypes::MOCK_1, CRE::OR, MockTypes::MOCK_2, CRE::CLOSE, CRE::AND, MockTypes::MOCK_2});

    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, Depth1NotEvaluated) {
    EXPECT_CALL(*cap_1, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, lazy_check_support()).Times(0);
    EXPECT_CALL(*cap_3, lazy_check_support()).Times(0);

    CRE cre({MockTypes::MOCK_1, CRE::OR, CRE::OPEN, MockTypes::MOCK_2, CRE::AND, MockTypes::MOCK_3, CRE::CLOSE});

    EXPECT_TRUE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, Depth2NotEvaluated) {
    EXPECT_CALL(*cap_1, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_3, lazy_check_support()).Times(0);

    CRE cre({CRE::NOT,
             MockTypes::MOCK_1,
             CRE::OR,
             CRE::OPEN,
             CRE::NOT,
             MockTypes::MOCK_2,
             CRE::AND,
             CRE::OPEN,
             MockTypes::MOCK_3,
             CRE::CLOSE,
             CRE::CLOSE});

    EXPECT_FALSE(cre.check_compatibility(caps));
}

TEST_F(CREOperandsEvaluation, AllDepthNotEvaluated) {
    EXPECT_CALL(*cap_1, lazy_check_support()).Times(1).WillOnce(::testing::Return(true));
    EXPECT_CALL(*cap_2, lazy_check_support()).Times(0);
    EXPECT_CALL(*cap_3, lazy_check_support()).Times(0);

    CRE cre({CRE::NOT,
             MockTypes::MOCK_1,
             CRE::AND,
             CRE::OPEN,
             MockTypes::MOCK_2,
             CRE::AND,
             CRE::OPEN,
             MockTypes::MOCK_3,
             CRE::CLOSE,
             CRE::CLOSE});

    EXPECT_FALSE(cre.check_compatibility(caps));
}
