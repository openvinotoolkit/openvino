// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_common.hpp"

using get_property_params = std::tuple<std::string,  // Get Property Name
                                       bool>;        // Throw exception

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
        ON_CALL(*m_plugin, get_property(StrEq("PERF_COUNT"), _)).WillByDefault(Return(true));
    }
};

TEST_P(GetPropertyTest, GetPropertyTestCase) {
    ov::AnyMap options = {};
    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->get_property(m_property_name, options));
    } else {
        ov::Any value;
        OV_ASSERT_NO_THROW(value = m_plugin->get_property(m_property_name, options));
        if (m_property_name == ov::device::full_name.name()) {
            EXPECT_EQ(value.as<std::string>(), "BATCH");
            return;
        }
    }
}

const std::vector<get_property_params> get_property_params_test = {
    get_property_params{ov::auto_batch_timeout.name(), false},
    get_property_params{ov::device::priorities.name(), true},
    get_property_params{ov::cache_dir.name(), true},
    get_property_params{ov::hint::performance_mode.name(), true},
    get_property_params{ov::enable_profiling.name(), false},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         GetPropertyTest,
                         ::testing::ValuesIn(get_property_params_test),
                         GetPropertyTest::getTestCaseName);
