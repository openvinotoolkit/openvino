// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/section_id.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/common/cre.hpp"
#include "intel_npu/common/section_type.hpp"

using namespace intel_npu;

using testing::_;
using SectionIdTest = ::testing::Test;

TEST_F(SectionIdTest, CreateAndGetID) {
    ASSERT_EQ(SectionID(0).get_id(), 0);
    ASSERT_EQ(SectionID(std::numeric_limits<uint16_t>::max()).get_id(), std::numeric_limits<uint16_t>::max());
}

TEST_F(SectionIdTest, EqualOperator) {
    ASSERT_TRUE(SectionID(0) == SectionID(0));
    ASSERT_FALSE(SectionID(0) == SectionID(1));
}

TEST_F(SectionIdTest, DifferentOperator) {
    ASSERT_FALSE(SectionID(0) != SectionID(0));
    ASSERT_TRUE(SectionID(0) != SectionID(1));
}

TEST_F(SectionIdTest, LowerOperator) {
    ASSERT_FALSE(SectionID(1) < SectionID(0));
    ASSERT_FALSE(SectionID(1) < SectionID(1));
    ASSERT_TRUE(SectionID(0) < SectionID(1));
}

TEST_F(SectionIdTest, ToString) {
    ASSERT_EQ(SectionID(0).to_string(), "0");
    ASSERT_EQ(SectionID(std::numeric_limits<uint16_t>::max()).to_string(),
              std::to_string(std::numeric_limits<uint16_t>::max()));
}

TEST_F(SectionIdTest, ValidFromString) {
    ASSERT_EQ(SectionID::from_string("0"), SectionID(0));
    ASSERT_EQ(SectionID::from_string(std::to_string(std::numeric_limits<uint16_t>::max())),
              SectionID(std::numeric_limits<uint16_t>::max()));
}

TEST_F(SectionIdTest, InvalidFromString) {
    OV_EXPECT_THROW(SectionID::from_string(""), ov::Exception, _);
    OV_EXPECT_THROW(SectionID::from_string("_0"), ov::Exception, _);
    OV_EXPECT_THROW(SectionID::from_string("0.0"), ov::Exception, _);
    OV_EXPECT_THROW(SectionID::from_string("section_id"), ov::Exception, _);
    OV_EXPECT_THROW(SectionID::from_string("100000000000000000"), ov::Exception, _);
}

TEST_F(SectionIdTest, IsSectionId) {
    const auto section_type = std::make_shared<SectionType>(SectionTypeCode::MANIFEST);
    const auto section_id = std::make_shared<SectionID>(0);
    const auto cre_special_token = std::make_shared<CRESpecialToken>(CRESpecialTokenCode::AND);
    ASSERT_FALSE(is_section_id(section_type));
    ASSERT_TRUE(is_section_id(section_id));
    ASSERT_FALSE(is_section_id(cre_special_token));
}
