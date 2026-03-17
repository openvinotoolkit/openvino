// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "expressions.hpp"
#include "intel_npu/common/cre.hpp"
#include "intel_npu/common/static_capability.hpp"

using namespace intel_npu;

using CREParams = std::tuple<std::string, std::vector<CRE::Token>, std::vector<uint16_t>, bool>;

// try to always have the order changed of written sections
class CREUnitTests : public ::testing::TestWithParam<CREParams> {
protected:
    void SetUp() override {
        std::tie(std::ignore, expression, section_types, is_compatible) = GetParam();

        cre = CRE(expression);

        for (const auto& token : section_types) {
            // compile-time assert if an unknown token is used for StaticCapability?
            capabilities[token] = std::make_shared<StaticCapability>(token);
        }
    }
    std::vector<CRE::Token> expression;
    CRE cre;
    std::vector<uint16_t> section_types;
    std::unordered_map<CRE::Token, std::shared_ptr<ICapability>> capabilities;
    bool is_compatible;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CREParams> obj) {
        bool is_compatible;
        std::tie(std::ignore, std::ignore, std::ignore, is_compatible) = obj.param;
        std::string compatibility_str = is_compatible ? "compatible" : "incompatible";
        return std::get<0>(obj.param) + "_" + compatibility_str;
    }
};

TEST_P(CREUnitTests, check_compatibility) {
    EXPECT_EQ(cre.check_compatibility(capabilities), is_compatible);
}

// to test later: order of serialization vs order of expression
// what if there is something in the expression in the capabilities found and vice versa
