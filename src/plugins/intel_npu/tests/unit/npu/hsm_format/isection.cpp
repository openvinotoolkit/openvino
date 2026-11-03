// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "utils.hpp"

using namespace intel_npu;
using testing::_;

namespace {

class TestSection : public ISection {
public:
    TestSection(const bool is_inline)
        : ISection(SectionTypeCode::BATCH_SIZE, ov::runtime::hsm::SectionAlignment{10, 11}, is_inline) {}
};

}  // namespace

class ISectionTest : public testing::Test {
public:
    ISectionTest() : section(true) {}

    TestSection section;
};

TEST_F(ISectionTest, GetType) {
    ASSERT_EQ(section.get_type(), SectionType(SectionTypeCode::BATCH_SIZE));
}

TEST_F(ISectionTest, GetId) {
    ASSERT_EQ(section.get_id(), std::nullopt);
}

TEST_F(ISectionTest, GetAlignment) {
    ASSERT_EQ(section.get_alignment().offset_align, 10);
    ASSERT_EQ(section.get_alignment().size_align, 11);
}

TEST_F(ISectionTest, IsInline) {
    ASSERT_TRUE(section.is_inline());
    section = TestSection(false);
    ASSERT_FALSE(section.is_inline());
}

TEST_F(ISectionTest, GetIndividualRequirements) {
    ASSERT_EQ(section.get_individual_compatibility_requirements(), std::nullopt);
}

TEST_F(ISectionTest, GetCRETokens) {
    ASSERT_TRUE(section.get_compatibility_requirements_subexpression({}).empty());
}

TEST_F(ISectionTest, EncodeNotImplemented) {
    std::ostringstream stream;
    OV_EXPECT_THROW(section.encode(), ov::NotImplemented, _);
    OV_EXPECT_THROW(section.encode(create_generic_section_sink(stream)), ov::NotImplemented, _);
}
