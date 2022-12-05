// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/properties.hpp"
#include "log/log.hpp"

using namespace ov::intel_gna;

namespace test {

using GnaLogTestParams = std::tuple<ov::log::Level, ov::log::Level>;

class GnaLogTest : public ::testing::TestWithParam<GnaLogTestParams> {
    protected:
        void SetUp() override  {}
    public:
        static std::string GetTestCaseName(const testing::TestParamInfo<GnaLogTestParams>& obj) {
            ov::log::Level log_level, message_level;
            std::tie(log_level, message_level) = obj.param;
            std::ostringstream result;
            result << "LogLevel=" << log_level;
            result << "_MsgLevel=" << message_level;
            return result.str();
        }
};

TEST_P(GnaLogTest, LogLevel) {
    std::string test_message = "Test message";
    std::stringstream expected_message;
    std::stringstream buffer;
    std::streambuf *sbuf = std::cout.rdbuf();
    std::streambuf *ebuf = std::cerr.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
    std::cerr.rdbuf(buffer.rdbuf());

    ov::log::Level log_level, message_level;
    std::tie(log_level, message_level) = GetParam();

    const auto prev_log_level = GnaLog::get_log_level();
    GnaLog::set_log_level(log_level);

    switch (message_level) {
    case ov::log::Level::ERR :
        log::error()  << test_message << std::endl;
        break;

    case ov::log::Level::WARNING :
        log::warning()  << test_message << std::endl;
        break;

    case ov::log::Level::INFO :
        log::info()  << test_message << std::endl;
        break;

    case ov::log::Level::DEBUG :
        log::debug()  << test_message << std::endl;
        break;

    case ov::log::Level::TRACE :
        log::trace()  << test_message << std::endl;
    break;

    default:
        break;
    }

    expected_message << "[" << message_level << "] " << test_message << std::endl;
    if (message_level <= log_level) {
        EXPECT_TRUE(buffer.str() == expected_message.str());
    } else {
        EXPECT_TRUE(buffer.str().empty());
    }

    std::cout.rdbuf(sbuf);
    std::cerr.rdbuf(ebuf);
    GnaLog::set_log_level(prev_log_level);
}

INSTANTIATE_TEST_SUITE_P(smoke_GnaLogTest,
                         GnaLogTest,
                         ::testing::Combine(
                            ::testing::ValuesIn({ov::log::Level::NO,
                                                 ov::log::Level::ERR,
                                                 ov::log::Level::WARNING,
                                                 ov::log::Level::INFO,
                                                 ov::log::Level::DEBUG,
                                                 ov::log::Level::TRACE}),
                            ::testing::ValuesIn({ov::log::Level::ERR,
                                                 ov::log::Level::WARNING,
                                                 ov::log::Level::INFO,
                                                 ov::log::Level::DEBUG,
                                                 ov::log::Level::TRACE})),
                        GnaLogTest::GetTestCaseName);
} //namespace test