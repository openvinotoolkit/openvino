// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <future>

#include "utils/log_util.hpp"
using ::testing::_;
using namespace ov::mock_auto_plugin;
// disable using windows.h
#if 0
#    if defined(_WIN32)
#        include <windows.h>
#    elif defined(__linux__)
#        include <stdlib.h>
#    elif defined(__APPLE__)
#        include <stdlib.h>
#    else
#    endif
#endif

MockLog* MockLog::m_mocklog = NULL;
using ConfigParams = std::tuple<std::string,  // logLevel
                                std::string,  // envlogLevel
                                int           //  expectCallNum
                                >;
class LogUtilsTest : public ::testing::TestWithParam<ConfigParams> {
public:
    std::string _logLevel;
    std::string _envLogLevel;
    int _expectCallNum;

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConfigParams> obj) {
        std::string logLevel;
        std::string envLogLevel;
        int expectCallNum;
        std::tie(logLevel, envLogLevel, expectCallNum) = obj.param;
        std::ostringstream result;
        result << "logLevel_" << logLevel << "_expectCallNum_" << expectCallNum << "envlogLevel" << envLogLevel;
        return result.str();
    }

#if 0
    void SetTestEnv(std::string key, std::string value) {
#    ifdef WIN32
        SetEnvironmentVariable(key.c_str(), value.c_str());
#    elif defined(__linux__)
        ::setenv(key.c_str(), value.c_str(), true);
#    elif defined(__APPLE__)
        ::setenv(key.c_str(), value.c_str(), true);
#    else
#    endif
    }
#endif
    void SetUp() override {
        std::tie(_logLevel, _envLogLevel, _expectCallNum) = this->GetParam();
    }

    void TearDown() override {
        MockLog::release();
    }

    void printLog() {
        LOG_TRACE(true, "test", "TRACE");
        LOG_DEBUG("DEBUG");
        LOG_INFO("INFO");
        LOG_WARNING("WARNING");
        LOG_ERROR("ERROR");
        LOG_TRACE(true, "test", "%s", "TRACE");
        LOG_DEBUG("%s", "DEBUG");
        LOG_INFO("%s", "INFO");
        LOG_WARNING("%s", "WARNING");
        LOG_ERROR("%s", "ERROR");
    }
};

TEST_P(LogUtilsTest, set_log_level) {
    EXPECT_CALL(*(HLogger), print(_)).Times(_expectCallNum);
    set_log_level(_logLevel);
    printLog();
}

TEST_P(LogUtilsTest, INFO_RUN) {
    set_log_level(_logLevel);
    int a = 0;
    INFO_RUN([&a]() {
        a++;
    });
    if (_logLevel == "LOG_INFO" || _logLevel == "LOG_DEBUG" || _logLevel == "LOG_TRACE") {
        EXPECT_EQ(a, 1);
    } else {
        EXPECT_EQ(a, 0);
    }
}

TEST_P(LogUtilsTest, DEBUG_RUN) {
    set_log_level(_logLevel);
    int a = 0;
    DEBUG_RUN([&a]() {
        a++;
    });
    if (_logLevel == "LOG_DEBUG" || _logLevel == "LOG_TRACE") {
        EXPECT_EQ(a, 1);
    } else {
        EXPECT_EQ(a, 0);
    }
}

#if 0
TEST_P(LogUtilsTest, setEnvNotAffectset_log_level) {
    EXPECT_CALL(*(HLogger), print(_)).Times(_expectCallNum);
    set_log_level(_logLevel);
    SetTestEnv("OPENVINO_LOG_LEVEL", "3");
    printLog();
}
#endif

// can not test ENV case. because of the ENV variable is readed at the
// beginning of test application and modify it in runtime is not valid
// still need to test it in different platform manully
// TEST_P(LogUtilsTest, setEnvLogLevel) {
//    SetTestEnv("AUTO_LOG_LEVEL", _envLogLevel);
//    EXPECT_CALL(*(HLogger), print(_)).Times(_expectCallNum);
//    printLog();
//}
//

TEST(smoke_Auto_BehaviorTests, LogUtilsSingleton) {
    std::vector<std::future<void>> futureVect;
    std::shared_ptr<Log> instanceVector[20];
    for (unsigned int i = 0; i < 20; i++) {
        auto future = std::async(std::launch::async, [&instanceVector, i] {
            instanceVector[i] = Log::instance();
        });
        futureVect.push_back(std::move(future));
    }

    for (auto& future : futureVect) {
        future.wait();
    }

    for (unsigned int i = 0; i < 19; i++) {
        EXPECT_NE(instanceVector[i].get(), nullptr);
        EXPECT_EQ(instanceVector[i].get(), instanceVector[i + 1].get());
    }
}

const std::vector<ConfigParams> testConfigs = {ConfigParams{"LOG_NONE", "0", 0},
                                               ConfigParams{"LOG_NONE", "1", 0},
                                               ConfigParams{"LOG_ERROR", "2", 2},
                                               ConfigParams{"LOG_WARNING", "3", 4},
                                               ConfigParams{"LOG_INFO", "4", 6},
                                               ConfigParams{"LOG_DEBUG", "5", 8},
                                               ConfigParams{"LOG_TRACE", "6", 10}};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         LogUtilsTest,
                         ::testing::ValuesIn(testConfigs),
                         LogUtilsTest::getTestCaseName);
