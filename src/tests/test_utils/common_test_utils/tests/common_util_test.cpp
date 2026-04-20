// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>

#include "common_test_utils/common_utils.hpp"
#include "openvino/util/common_util.hpp"

namespace ov::test {

using CommonUtilsTest = testing::Test;

TEST_F(CommonUtilsTest, ltrim) {
    EXPECT_EQ("space test", util::ltrim("   space test"));
    EXPECT_EQ("tab,linefeed test", util::ltrim("\t\ntab,linefeed test"));
    EXPECT_EQ("", util::ltrim(""));
    EXPECT_EQ("", util::ltrim(" \r  "));
    EXPECT_EQ("carriage test ", util::ltrim("\rcarriage test "));
    EXPECT_EQ("vertical tab test ", util::ltrim("\v vertical tab test "));
}

TEST_F(CommonUtilsTest, rtrim) {
    EXPECT_EQ("space test", util::rtrim("space test   "));
    EXPECT_EQ("\ttab,linefeed test", util::rtrim("\ttab,linefeed test\t\n"));
    EXPECT_EQ("", util::rtrim(""));
    EXPECT_EQ("", util::rtrim(" \r  "));
    EXPECT_EQ(" carriage test", util::rtrim(" carriage test \r"));
    EXPECT_EQ("\v vertical tab test", util::rtrim("\v vertical tab test \v"));
}

TEST_F(CommonUtilsTest, trim) {
    EXPECT_EQ("space test", util::trim("   space test   "));
    EXPECT_EQ("tab,linefeed test", util::trim("\t\ntab,linefeed test\t\n"));
    EXPECT_EQ("", util::trim(""));
    EXPECT_EQ("", util::trim(" \r  "));
    EXPECT_EQ("carriage test", util::trim("\rcarriage test \r"));
    EXPECT_EQ("vertical tab test", util::trim("\v vertical tab test \v"));
}

TEST_F(CommonUtilsTest, parse_views_into_container_empty_fields) {
    std::vector<std::string_view> res;
    util::view_transform(" 1, , 2, 3", std::back_inserter(res), ",");
    EXPECT_EQ((std::vector<std::string_view>{" 1"," ", " 2", " 3"}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_trailing_separator) {
    std::vector<std::string_view> res;
    util::view_transform("1,2,", std::back_inserter(res), ",");
    EXPECT_EQ((std::vector<std::string_view>{"1", "2", ""}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_empty_input) {
    std::vector<std::string_view> res;
    util::view_transform("", std::back_inserter(res), ",");
    EXPECT_TRUE(res.empty());
}

TEST_F(CommonUtilsTest, parse_views_into_container_custom_check) {
    std::vector<std::string_view> res;
    util::view_transform_if(" 1,;2,; 3,; ", std::back_inserter(res), ",;", [](auto&& field) {
        OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1,;2,; 3,; \" is incorrect");
        return true;
    });
    EXPECT_EQ((std::vector<std::string_view>{" 1", "2", " 3", " "}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_custom_check_fail) {
    std::vector<std::string> res;
    EXPECT_THROW(util::view_transform_if(" 1,, 2, 3, ", std::back_inserter(res), ",", [](auto&& field) {
                     OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1, , 2, 3, \" is incorrect");
                     return true;
                 }, [](auto&& field) {
                     return std::string{field};
                 }  ),
                 ov::AssertFailure);
}

TEST_F(CommonUtilsTest, parse_views_into_container_arithmetic) {
    std::vector<int> res{5,5,5};
    util::view_transform("1,2,3", std::back_inserter(res), ",",
    [](auto&& field) { return util::view_to_number<int>(field).value_or(0); });
    EXPECT_EQ((std::vector<int>{5,5,5,1, 2, 3}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_arithmetic_fail) {
    std::vector<int> res(3);
    ASSERT_NO_THROW(util::view_transform("1, a, 3 ", res.begin(),",",
    [](auto&& field) { return util::view_to_number<int>(field).value_or(0); }));
    EXPECT_EQ((std::vector<int>{1, 0, 0}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_arithmetic_custom_check_fail) {
    std::vector<int> res;
    EXPECT_THROW(util::view_transform("1,,3", std::back_inserter(res), ",", [](auto&& field) {
                     OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1, a, 3 \" is incorrect");
                     return true;
                 }),
                 ov::AssertFailure);
}

TEST_F(CommonUtilsTest, split_to_views_default_separator) {
    EXPECT_EQ((std::vector<std::string_view>{"1", "2", "3"}), util::split("1,2,3"));
}

TEST_F(CommonUtilsTest, split_to_views_custom_separator) {
    EXPECT_EQ((std::vector<std::string_view>{"1", "2", "3"}), util::split("1;2;3", ";"));
}

TEST_F(CommonUtilsTest, split_to_views_custom_check) {
    EXPECT_EQ((std::vector<std::string_view>{"1", "2", "3"}),
              util::split("1;2;3", ";", [](auto&& field) {
                  OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1;2;3 \" is incorrect");
                  return true;
              }));
}

TEST_F(CommonUtilsTest, split_to_views_custom_check_fail) {
    EXPECT_THROW(util::split(
                     "1;;3",
                     ";",
                     [](auto&& field) {
                         OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1;;3 \" is incorrect");
                         return true;
                     }),
                 ov::AssertFailure);
}

TEST_F(CommonUtilsTest, split_to_views_lower) {
    const std::string test_str{"Test1,Test2,Test3"};
    EXPECT_EQ((std::vector<std::string_view>{"test1", "test2", "test3"}), util::split(util::to_lower(test_str)));
}

TEST_F(CommonUtilsTest, split_empty_strings) {
    using testing::ElementsAre;
    EXPECT_THAT(util::split2(""), testing::IsEmpty());
    EXPECT_THAT(util::split2(","), ElementsAre("", ""));
    EXPECT_THAT(util::split2(",,"), ElementsAre("", "", ""));
    EXPECT_THAT(util::split2("test,"), ElementsAre("test", ""));
    EXPECT_THAT(util::split2(",test"), ElementsAre("", "test"));
}

TEST_F(CommonUtilsTest, view_to_integral_number){
    EXPECT_EQ(123, util::view_to_number<int>("123").value());
    EXPECT_EQ(-123, util::view_to_number<int>("-123").value());
    EXPECT_EQ(123, util::view_to_number<int>("123abc").value());
}

TEST_F(CommonUtilsTest, view_to_number_invalid_input){
    EXPECT_EQ(0, util::view_to_number<int>("abc").value_or(0));
    EXPECT_EQ(0, util::view_to_number<int>("").value_or(0));
}

TEST_F(CommonUtilsTest, view_to_floating_point_number){
    EXPECT_FLOAT_EQ(123.456f, util::view_to_number<float>("123.456").value_or(0));
    EXPECT_DOUBLE_EQ(123.456, util::view_to_number<double>("123.456").value_or(0));
    EXPECT_DOUBLE_EQ(-123.456, util::view_to_number<double>("-123.456").value_or(0));
    EXPECT_DOUBLE_EQ(0, util::view_to_number<double>("abc").value_or(0));
    EXPECT_DOUBLE_EQ(0, util::view_to_number<double>("").value_or(0));
}

TEST_F(CommonUtilsTest, ends_with) {
    EXPECT_TRUE(util::ends_with("test.cpp", ".cpp"));
    EXPECT_TRUE(util::ends_with("test.cpp", "cpp"));
    EXPECT_TRUE(util::ends_with("test.cpp", ""));
    EXPECT_FALSE(util::ends_with("test.cpp", ".h"));
    EXPECT_FALSE(util::ends_with("test.cpp", "cpp "));
    EXPECT_FALSE(util::ends_with("d", ".cpp"));
    EXPECT_FALSE(util::ends_with("test.bin.bak", ".bin"));
}
}  // namespace ov::test
