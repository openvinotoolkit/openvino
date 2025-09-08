// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <regex>

#include "utils/log_util.hpp"
#include "common_test_utils/test_assertions.hpp"

using namespace ov::mock_auto_plugin;
using ::testing::_;
class LogUtilsFormatTest : public ::testing::Test {
public:
    void SetUp() override {
        set_log_level("LOG_DEBUG");
    }

    void TearDown() override {
        MockLog::release();
    }

    void traceCallStacksTest() {
        TraceCallStacks("test");
    }
};

TEST_F(LogUtilsFormatTest, callStacksTest) {
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    OV_ASSERT_NO_THROW(traceCallStacksTest());
}

TEST_F(LogUtilsFormatTest, format_s) {
    std::string printResult = "";
    std::string pattern{"abc"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%sabc", "DEBUG");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}
TEST_F(LogUtilsFormatTest, format_d) {
    std::string printResult = "";
    std::string pattern{"abc"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%dabc", -1);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_ld) {
    std::string printResult = "";
    std::string pattern{"abc"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%ldabc", -3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_u) {
    std::string printResult = "";
    std::string pattern{"abc"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%uabc", 1);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_lu) {
    std::string printResult = "";
    std::string pattern{"abc"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%luabc", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_s_d_ld_u_lu) {
    std::string printResult = "";
    std::string pattern{"abc"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%s,%d,%ld,%u,%lu,abc", "DEBUG", -1, -3, 1, 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_s_d_ld_u_lu2) {
    std::string printResult = "";
    std::string pattern{"abc"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%s%d%ld%u%luabc", "DEBUG", -1, -3, 1, 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_lf) {
    std::string printResult = "";
    std::string pattern{"abc"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%lfabc", 1.33);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_p) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%p", MockLog::m_mocklog);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_x) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%x", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_X) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%X", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_o) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%o", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_e) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%e", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_E) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%E", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_f) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%f", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_F) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%F", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_g) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%g", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_G) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%G", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_a) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%a", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_A) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%A", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_c) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%c", 3);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_n) {
    int num = 0;
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%n", &num);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format__) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%%");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_s__) {
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%s%%", "DEBUG");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_dn) {
    int num = 0;
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("%d%n", num, &num);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, format_ccccdn) {
    int num = 0;
    std::string printResult = "";
    std::string pattern{"not valid"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("cccc%d%n", num, &num);
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, logPrintFormat_error) {
    std::string printResult = "";
    std::string pattern{"\\[[0-9]+:[0-9]+:[0-9]+\\.[0-9]+\\]ERROR\\[.+:[0-9]+\\].*"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
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
        printResult = stream.str();
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
        printResult = stream.str();
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
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_DEBUG("test");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}

TEST_F(LogUtilsFormatTest, logPrintFormat_trace) {
    set_log_level("LOG_TRACE");
    std::string printResult = "";
    std::string pattern{"\\[[0-9]+:[0-9]+:[0-9]+\\.[0-9]+\\]T\\[.+:[0-9]+\\].*"};
    std::regex regex(pattern);
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) {
        printResult = stream.str();
    });
    EXPECT_CALL(*(HLogger), print(_)).Times(1);
    LOG_TRACE(true, "test", "TRACE");
    EXPECT_TRUE(std::regex_search(printResult, regex));
}