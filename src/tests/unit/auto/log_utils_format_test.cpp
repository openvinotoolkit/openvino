// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "utils/log_util.hpp"
#include <regex>
using namespace MockMultiDevice;
using ::testing::_;
class LogUtilsFormatTest : public ::testing::Test {
public:
    void SetUp() override {
        setLogLevel("LOG_DEBUG");
    }

    void TearDown() override {
        MockLog::Release();
    }
};

TEST_F(LogUtilsFormatTest, format_s) {
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    ASSERT_NO_THROW(LOG_DEBUG("%s", "DEBUG"));
}
TEST_F(LogUtilsFormatTest, format_d) {
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    ASSERT_NO_THROW(LOG_DEBUG("%d", -1));
}

TEST_F(LogUtilsFormatTest, format_ld) {
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    ASSERT_NO_THROW(LOG_DEBUG("%ld", -3));
}

TEST_F(LogUtilsFormatTest, format_u) {
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    ASSERT_NO_THROW(LOG_DEBUG("%u", 1));
}

TEST_F(LogUtilsFormatTest, format_lu) {
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    ASSERT_NO_THROW(LOG_DEBUG("%lu", 3));
}

TEST_F(LogUtilsFormatTest, format_s_d_ld_u_lu) {
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    ASSERT_NO_THROW(LOG_DEBUG("%s,%d,%ld,%u,%lu", "DEBUG", -1, -3, 1, 3));
}

TEST_F(LogUtilsFormatTest, format_s_d_ld_u_lu2) {
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    ASSERT_NO_THROW(LOG_DEBUG("%s%d%ld%u%lu", "DEBUG", -1, -3, 1, 3));
}

TEST_F(LogUtilsFormatTest, format_p) {
    ASSERT_THROW(LOG_DEBUG("%p", MockLog::_mockLog),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_x) {
    ASSERT_THROW(LOG_DEBUG("%x", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_X) {
    ASSERT_THROW(LOG_DEBUG("%X", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_o) {
    ASSERT_THROW(LOG_DEBUG("%o", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_e) {
    ASSERT_THROW(LOG_DEBUG("%e", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_E) {
    ASSERT_THROW(LOG_DEBUG("%E", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_f) {
    ASSERT_THROW(LOG_DEBUG("%f", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_F) {
    ASSERT_THROW(LOG_DEBUG("%F", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_g) {
    ASSERT_THROW(LOG_DEBUG("%g", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_G) {
    ASSERT_THROW(LOG_DEBUG("%G", 3),  std::exception);
}


TEST_F(LogUtilsFormatTest, format_a) {
    ASSERT_THROW(LOG_DEBUG("%a", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_A) {
    ASSERT_THROW(LOG_DEBUG("%A", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_c) {
    ASSERT_THROW(LOG_DEBUG("%c", 3),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_n) {
    int num = 0;
    ASSERT_THROW(LOG_DEBUG("%n", &num),  std::exception);
}

TEST_F(LogUtilsFormatTest, format__) {
    ASSERT_THROW(LOG_DEBUG("%%"),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_s__) {
    ASSERT_THROW(LOG_DEBUG("%s%%", "DEBUG"),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_dn) {
    int num = 0;
    ASSERT_THROW(LOG_DEBUG("%d%n", num, &num),  std::exception);
}

TEST_F(LogUtilsFormatTest, format_ccccdn) {
    int num = 0;
    ASSERT_THROW(LOG_DEBUG("cccc%d%n", num, &num),  std::exception);
}

TEST_F(LogUtilsFormatTest, logPrintFormat_error) {
    std::string printResult = "";
    std::string pattern{"\\[[0-9]+:[0-9]+:[0-9]+\\.[0-9]+\\]ERROR\\[.+:[0-9]+\\].*"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
            printResult =  stream.str();
            });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_ERROR("test");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, logPrintFormat_warning) {
    std::string printResult = "";
    std::string pattern{"\\[[0-9]+:[0-9]+:[0-9]+\\.[0-9]+\\]W\\[.+:[0-9]+\\].*"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
            printResult =  stream.str();
            });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_WARNING("test");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, logPrintFormat_info) {
    std::string printResult = "";
    std::string pattern{"\\[[0-9]+:[0-9]+:[0-9]+\\.[0-9]+\\]I\\[.+:[0-9]+\\].*"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
            printResult =  stream.str();
            });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_INFO("test");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, logPrintFormat_debug) {
    std::string printResult = "";
    std::string pattern{"\\[[0-9]+:[0-9]+:[0-9]+\\.[0-9]+\\]D\\[.+:[0-9]+\\].*"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
            printResult =  stream.str();
            });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("test");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, logPrintFormat_trace) {
    setLogLevel("LOG_TRACE");
    std::string printResult = "";
    std::string pattern{"\\[[0-9]+:[0-9]+:[0-9]+\\.[0-9]+\\]T\\[.+:[0-9]+\\].*"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
            printResult =  stream.str();
            });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_TRACE(true, "test", "TRACE");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

