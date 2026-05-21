// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "intel_npu/common/cre.hpp"
#include "intel_npu/common/static_capability.hpp"

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
            supported_capabilities[token] = std::make_shared<StaticCapability>(token);
        }
    }

    CRE cre;
    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> supported_capabilities;
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
    EXPECT_EQ(cre.get_expression_length(), 2);
    EXPECT_EQ(cre.get_expression(), (std::vector<CRE::Token>{CRE::AND, CRE::ELF_SCHEDULE}));
}

TEST_F(CREAppendSingleToken, AppendMultipleValidTokensAccumulates) {
    CRE cre;
    cre.append_to_expression(CRE::ELF_SCHEDULE);
    cre.append_to_expression(CRE::BATCHING);
    cre.append_to_expression(CRE::WEIGHTS_SEPARATION);
    EXPECT_EQ(cre.get_expression_length(), 4);
    EXPECT_EQ(cre.get_expression(),
              (std::vector<CRE::Token>{CRE::AND, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION}));
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

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[CRE::ELF_SCHEDULE] = std::make_shared<StaticCapability>(CRE::ELF_SCHEDULE);
    caps[CRE::BATCHING] = std::make_shared<StaticCapability>(CRE::BATCHING);
    EXPECT_TRUE(cre.check_compatibility(caps));

    caps.erase(CRE::BATCHING);
    EXPECT_FALSE(cre.check_compatibility(caps));
}

using CREAppendToken = ::testing::Test;

TEST_F(CREAppendToken, AppendEmptyVector) {
    CRE cre;
    cre.append_to_expression(std::vector<CRE::Token>{});
    EXPECT_EQ(cre.get_expression_length(), 1);
    EXPECT_EQ(cre.get_expression(), (std::vector<CRE::Token>{CRE::AND}));
}

TEST_F(CREAppendToken, AppendSubexpressionTokens) {
    CRE cre;
    cre.append_to_expression(
        std::vector<CRE::Token>{CRE::OPEN, CRE::OR, CRE::BATCHING, CRE::WEIGHTS_SEPARATION, CRE::CLOSE});

    EXPECT_EQ(
        cre.get_expression(),
        (std::vector<CRE::Token>{CRE::AND, CRE::OPEN, CRE::OR, CRE::BATCHING, CRE::WEIGHTS_SEPARATION, CRE::CLOSE}));
}

TEST_F(CREAppendToken, MixedAppend) {
    CRE cre;
    cre.append_to_expression(CRE::ELF_SCHEDULE);
    cre.append_to_expression(
        std::vector<CRE::Token>{CRE::OPEN, CRE::OR, CRE::BATCHING, CRE::WEIGHTS_SEPARATION, CRE::CLOSE});

    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> caps;
    caps[CRE::ELF_SCHEDULE] = std::make_shared<StaticCapability>(CRE::ELF_SCHEDULE);
    caps[CRE::BATCHING] = std::make_shared<StaticCapability>(CRE::BATCHING);
    EXPECT_TRUE(cre.check_compatibility(caps));

    caps.erase(CRE::ELF_SCHEDULE);
    EXPECT_FALSE(cre.check_compatibility(caps));
}
