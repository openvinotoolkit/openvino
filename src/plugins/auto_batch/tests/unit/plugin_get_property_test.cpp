// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_common.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;

using namespace ov::mock_autobatch_plugin;

using get_property_params = std::tuple<std::string,  // Get Property Name
                                       bool>;        // Throw exception

const char supported_metric[] = "SUPPORTED_METRICS FULL_DEVICE_NAME SUPPORTED_CONFIG_KEYS";
const char supported_config_keys[] = "AUTO_BATCH_DEVICE_CONFIG MULTI_DEVICE_PRIORITIES AUTO_BATCH_TIMEOUT CACHE_DIR";

class GetPropertyTest : public ::testing::TestWithParam<get_property_params> {
public:
    std::string m_property_name;
    bool m_throw_exception;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<get_property_params> obj) {
        std::string property_name;
        bool throw_exception;

        std::tie(property_name, throw_exception) = obj.param;
        std::string res = "";

        if (!property_name.empty()) {
            res += "GetProperty_" + property_name;
        }
        if (throw_exception)
            res += "_throw";
        return res;
    }

    void TearDown() override {
        m_plugin.reset();
    }

    void SetUp() override {
        std::tie(m_property_name, m_throw_exception) = this->GetParam();
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());

        ON_CALL(*m_plugin, get_property).WillByDefault([this](const std::string& name, const ov::AnyMap& arguments) {
            return m_plugin->Plugin::get_property(name, arguments);
        });
    }
};

TEST_P(GetPropertyTest, GetPropertyTestCase) {
    ov::AnyMap options = {};
    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->get_property(m_property_name, options));
    } else {
        ov::Any value;
        ASSERT_NO_THROW(value = m_plugin->get_property(m_property_name, options));
        if (m_property_name == METRIC_KEY(SUPPORTED_METRICS)) {
            EXPECT_EQ(value.as<std::string>(), supported_metric);
            return;
        }
        if (m_property_name == ov::device::full_name.name()) {
            EXPECT_EQ(value.as<std::string>(), "BATCH");
            return;
        }
        if (m_property_name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
            EXPECT_EQ(value.as<std::string>(), supported_config_keys);
            return;
        }
    }
}

const std::vector<get_property_params> get_property_params_test = {
    get_property_params{"AUTO_BATCH_TIMEOUT", false},
    get_property_params{"AUTO_BATCH_DEVICE_CONFIG", true},
    get_property_params{"CACHE_DIR", true},
    get_property_params{METRIC_KEY(SUPPORTED_METRICS), false},
    get_property_params{METRIC_KEY(SUPPORTED_CONFIG_KEYS), false},
    get_property_params{"CPU_THREADS_NUM", true},
    get_property_params{"PERFORMANCE_HINT", true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         GetPropertyTest,
                         ::testing::ValuesIn(get_property_params_test),
                         GetPropertyTest::getTestCaseName);
