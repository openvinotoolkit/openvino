#include <gtest/gtest.h>

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
    std::vector<std::string> res;
    util::parse_view_into_container(" 1, , 2, 3", res);
    EXPECT_EQ((std::vector<std::string>{" 1"," ", " 2", " 3"}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_custom_check) {
    std::vector<std::string> res;
    util::parse_view_into_container(" 1,;2,; 3,; ", res, ",;", [](auto&& field) {
        OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1,;2,; 3,; \" is incorrect");
    });
    EXPECT_EQ((std::vector<std::string>{" 1", "2", " 3", " "}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_custom_check_fail) {
    std::vector<std::string> res;
    EXPECT_THROW(util::parse_view_into_container(" 1,, 2, 3, ", res, ",", [](auto&& field) {
                     OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1, , 2, 3, \" is incorrect");
                 }),
                 ov::AssertFailure);
}

TEST_F(CommonUtilsTest, parse_views_into_container_arithmetic) {
    std::vector<int> res{5,5,5};
    util::parse_view_into_container("1,2,3", res);
    EXPECT_EQ((std::vector<int>{5,5,5,1, 2, 3}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_arithmetic_fail) {
    std::vector<int> res;
    ASSERT_NO_THROW(util::parse_view_into_container("1, a, 3 ", res));
    EXPECT_EQ((std::vector<int>{1, 0, 0}), res);
}

TEST_F(CommonUtilsTest, parse_views_into_container_arithmetic_custom_check_fail) {
    std::vector<int> res;
    EXPECT_THROW(util::parse_view_into_container("1,,3", res, ",", [](auto&& field) {
                     OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1, a, 3 \" is incorrect");
                 }),
                 ov::AssertFailure);
}

TEST_F(CommonUtilsTest, split_to_views_default_separator) {
    EXPECT_EQ((std::vector<std::string>{"1", "2", "3"}), util::split_to_views<std::vector<std::string>>("1,2,3"));
}

TEST_F(CommonUtilsTest, split_to_views_custom_separator) {
    EXPECT_EQ((std::vector<std::string>{"1", "2", "3"}), util::split_to_views<std::vector<std::string>>("1;2;3", ";"));
}

TEST_F(CommonUtilsTest, split_to_views_custom_check) {
    EXPECT_EQ((std::vector<std::string>{"1", "2", "3"}),
              util::split_to_views<std::vector<std::string>>("1;2;3", ";", [](auto&& field) {
                  OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1;2;3 \" is incorrect");
              }));
}

TEST_F(CommonUtilsTest, split_to_views_custom_check_fail) {
    EXPECT_THROW(util::split_to_views<std::vector<std::string>>(
                     "1;;3",
                     ";",
                     [](auto&& field) {
                         OPENVINO_ASSERT(!field.empty(), "Cannot get vector of fields! \" 1;;3 \" is incorrect");
                     }),
                 ov::AssertFailure);
}

}  // namespace ov::test
