// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_npu/common/cre.hpp"
#include "intel_npu/common/section_id.hpp"
#include "intel_npu/common/section_type.hpp"

using namespace intel_npu;

using CRESpecialTokenTest = ::testing::Test;

// TODO test to_string
TEST_F(CRESpecialTokenTest, CreateAllTokensUsingCorrectCode) {
    ASSERT_EQ(CRESpecialToken(CRESpecialTokenCode::AND).get_code(), CRESpecialTokenCode::AND);
    ASSERT_EQ(CRESpecialToken(CRESpecialTokenCode::OR).get_code(), CRESpecialTokenCode::OR);
    ASSERT_EQ(CRESpecialToken(CRESpecialTokenCode::NOT).get_code(), CRESpecialTokenCode::NOT);
    ASSERT_EQ(CRESpecialToken(CRESpecialTokenCode::OPEN).get_code(), CRESpecialTokenCode::OPEN);
    ASSERT_EQ(CRESpecialToken(CRESpecialTokenCode::CLOSE).get_code(), CRESpecialTokenCode::CLOSE);
}

TEST_F(CRESpecialTokenTest, EqualOperator) {
    ASSERT_TRUE(CRESpecialToken(CRESpecialTokenCode::AND) == CRESpecialToken(CRESpecialTokenCode::AND));
    ASSERT_FALSE(CRESpecialToken(CRESpecialTokenCode::AND) == CRESpecialToken(CRESpecialTokenCode::OR));
}

TEST_F(CRESpecialTokenTest, DifferentOperator) {
    ASSERT_FALSE(CRESpecialToken(CRESpecialTokenCode::AND) != CRESpecialToken(CRESpecialTokenCode::AND));
    ASSERT_TRUE(CRESpecialToken(CRESpecialTokenCode::AND) != CRESpecialToken(CRESpecialTokenCode::OR));
}

TEST_F(CRESpecialTokenTest, IsSpecialToken) {
    const auto section_type = std::make_shared<SectionType>(SectionTypeCode::MANIFEST);
    const auto section_id = std::make_shared<SectionID>(0);
    const auto cre_special_token = std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND);
    ASSERT_FALSE(is_cre_special_token(section_type));
    ASSERT_FALSE(is_cre_special_token(section_id));
    ASSERT_TRUE(is_cre_special_token(cre_special_token));
}
