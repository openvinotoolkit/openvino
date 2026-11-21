// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/major_minor_version.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"

using namespace intel_npu;
using testing::_;

namespace {}  // namespace

using MajorMinorVersionTest = testing::Test;

TEST_F(MajorMinorVersionTest, RetrieveValuesSetUsingCtor) {
    MajorMinorVersion version(0, 0);
    ASSERT_EQ(version.get_major(), 0);
    ASSERT_EQ(version.get_minor(), 0);

    const auto max_value = std::numeric_limits<uint16_t>::max();
    version = MajorMinorVersion(max_value, max_value);
    ASSERT_EQ(version.get_major(), max_value);
    ASSERT_EQ(version.get_minor(), max_value);
}

TEST_F(MajorMinorVersionTest, ComparisonOperators) {
    const MajorMinorVersion version1(0, 0);
    const MajorMinorVersion version2(1, 0);
    const MajorMinorVersion version3(0, 1);
    const MajorMinorVersion version4(1, 1);

    ASSERT_TRUE(version1 == version1);
    ASSERT_FALSE(version1 == version2);
    ASSERT_FALSE(version1 == version3);
    ASSERT_FALSE(version1 == version4);
    ASSERT_TRUE(version2 == version2);
    ASSERT_FALSE(version2 == version1);
    ASSERT_FALSE(version2 == version3);
    ASSERT_FALSE(version2 == version4);

    ASSERT_FALSE(version1 != version1);
    ASSERT_TRUE(version1 != version2);
    ASSERT_TRUE(version1 != version3);
    ASSERT_TRUE(version1 != version4);
    ASSERT_FALSE(version2 != version2);
    ASSERT_TRUE(version2 != version1);
    ASSERT_TRUE(version2 != version3);
    ASSERT_TRUE(version2 != version4);

    ASSERT_FALSE(version1 > version1);
    ASSERT_FALSE(version1 > version2);
    ASSERT_FALSE(version1 > version3);
    ASSERT_FALSE(version1 > version4);
    ASSERT_TRUE(version1 >= version1);
    ASSERT_FALSE(version1 >= version2);
    ASSERT_FALSE(version1 >= version3);
    ASSERT_FALSE(version1 >= version4);

    ASSERT_FALSE(version1 < version1);
    ASSERT_TRUE(version1 < version2);
    ASSERT_TRUE(version1 < version3);
    ASSERT_TRUE(version1 < version4);
    ASSERT_TRUE(version1 <= version1);
    ASSERT_TRUE(version1 <= version2);
    ASSERT_TRUE(version1 <= version3);
    ASSERT_TRUE(version1 <= version4);

    ASSERT_TRUE(version4 > version1);
    ASSERT_TRUE(version4 > version2);
    ASSERT_TRUE(version4 > version3);
    ASSERT_FALSE(version4 > version4);
    ASSERT_TRUE(version4 >= version1);
    ASSERT_TRUE(version4 >= version2);
    ASSERT_TRUE(version4 >= version3);
    ASSERT_TRUE(version4 >= version4);

    ASSERT_FALSE(version4 < version1);
    ASSERT_FALSE(version4 < version2);
    ASSERT_FALSE(version4 < version3);
    ASSERT_FALSE(version4 < version4);
    ASSERT_FALSE(version4 <= version1);
    ASSERT_FALSE(version4 <= version2);
    ASSERT_FALSE(version4 <= version3);
    ASSERT_TRUE(version4 <= version4);

    ASSERT_TRUE(version2 > version1);
    ASSERT_FALSE(version2 > version2);
    ASSERT_TRUE(version2 > version3);
    ASSERT_FALSE(version2 > version4);
    ASSERT_TRUE(version2 >= version1);
    ASSERT_TRUE(version2 >= version2);
    ASSERT_TRUE(version2 >= version3);
    ASSERT_FALSE(version2 >= version4);

    ASSERT_FALSE(version2 < version1);
    ASSERT_FALSE(version2 < version2);
    ASSERT_FALSE(version2 < version3);
    ASSERT_TRUE(version2 < version4);
    ASSERT_FALSE(version2 <= version1);
    ASSERT_TRUE(version2 <= version2);
    ASSERT_FALSE(version2 <= version3);
    ASSERT_TRUE(version2 <= version4);
}

TEST_F(MajorMinorVersionTest, ToString) {
    const auto max_value = std::numeric_limits<uint16_t>::max();
    const auto max_value_string = std::to_string(max_value);

    ASSERT_EQ(MajorMinorVersion(0, 0).to_string(), "0.0");
    ASSERT_EQ(MajorMinorVersion(max_value, 0).to_string(), max_value_string + ".0");
    ASSERT_EQ(MajorMinorVersion(0, max_value).to_string(), "0." + max_value_string);
    ASSERT_EQ(MajorMinorVersion(max_value, max_value).to_string(), max_value_string + "." + max_value_string);
}

TEST_F(MajorMinorVersionTest, FromStringValidValues) {
    const auto max_value = std::numeric_limits<uint16_t>::max();
    const auto max_value_string = std::to_string(max_value);

    ASSERT_EQ(MajorMinorVersion::from_string("0.0"), MajorMinorVersion(0, 0));
    ASSERT_EQ(MajorMinorVersion::from_string(max_value_string + ".0"), MajorMinorVersion(max_value, 0));
    ASSERT_EQ(MajorMinorVersion::from_string("0." + max_value_string), MajorMinorVersion(0, max_value));
    ASSERT_EQ(MajorMinorVersion::from_string(max_value_string + "." + max_value_string),
              MajorMinorVersion(max_value, max_value));
}

TEST_F(MajorMinorVersionTest, FromStringInvalidValues) {
    const auto max_value = std::numeric_limits<uint16_t>::max();

    OV_EXPECT_THROW(MajorMinorVersion::from_string("0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0."), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0.0."), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string(".0.0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0.0.0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string(".0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("."), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("-1.0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0.-1"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string(std::to_string(max_value + 1) + ".0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0." + std::to_string(max_value + 1)), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("a.0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0.a"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("$.0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0.$"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0..0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0a.0"), ov::Exception, _);
    OV_EXPECT_THROW(MajorMinorVersion::from_string("0.0a"), ov::Exception, _);
}

TEST_F(MajorMinorVersionTest, ParseDottedVersionUtilityValidStrings) {
    const auto max_value = std::numeric_limits<uint16_t>::max();
    const auto max_value_string = std::to_string(max_value);

    ASSERT_TRUE(parse_dotted_version("0.0", 2) == std::vector<uint16_t>({0, 0}));
    ASSERT_TRUE(parse_dotted_version(max_value_string + ".0", 2) == std::vector<uint16_t>({max_value, 0}));
    ASSERT_TRUE(parse_dotted_version("0." + max_value_string, 2) == std::vector<uint16_t>({0, max_value}));
    ASSERT_TRUE(parse_dotted_version(max_value_string + "." + max_value_string, 2) ==
                std::vector<uint16_t>({max_value, max_value}));

    ASSERT_TRUE(parse_dotted_version("", 0) == std::vector<uint16_t>({}));
    ASSERT_TRUE(parse_dotted_version("0", 1) == std::vector<uint16_t>({0}));
    ASSERT_TRUE(parse_dotted_version("0.1.2", 3) == std::vector<uint16_t>({0, 1, 2}));
}

TEST_F(MajorMinorVersionTest, ParseDottedVersionUtilityInvalidStrings) {
    const auto max_value = std::numeric_limits<uint16_t>::max();

    OV_EXPECT_THROW(parse_dotted_version("0.", 1), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("00", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.0.", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.0.", 3), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version(".0.0", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version(".0.0", 3), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version(".0", 1), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version(".0", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version(".", 0), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("-1.0", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.-1", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version(std::to_string(max_value + 1) + ".0", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0." + std::to_string(max_value + 1), 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("a.0", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.a", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("$.0", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.$", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0..0", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0..0", 3), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0a.0", 2), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.0a", 2), ov::Exception, _);
}

TEST_F(MajorMinorVersionTest, ParseDottedVersionUtilityUnexpectedValuesCount) {
    OV_EXPECT_THROW(parse_dotted_version("", 1), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.0", 1), ov::Exception, _);
    OV_EXPECT_THROW(parse_dotted_version("0.0", 3), ov::Exception, _);
}
