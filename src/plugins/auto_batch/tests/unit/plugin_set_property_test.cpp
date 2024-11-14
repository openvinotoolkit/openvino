// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_common.hpp"

using set_property_params = std::tuple<ov::AnyMap,  // Set Property
                                       bool>;

class SetPropertyTest : public ::testing::TestWithParam<set_property_params> {
public:
    ov::AnyMap m_properties;
    bool m_throw_exception;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;

public:
    static std::string getTestCaseName(testing::TestParamInfo<set_property_params> obj) {
        ov::AnyMap properties;
        bool throw_exception;

        std::tie(properties, throw_exception) = obj.param;
        std::string res = "";
        if (properties.size() > 0) {
            res += "SetProperty_";
            for (auto& it : properties) {
                res += it.first + "_" + it.second.as<std::string>() + "_";
            }
        }
        if (throw_exception)
            res += "_throw";
        return res;
    }

    void TearDown() override {
        m_plugin.reset();
    }

    void SetUp() override {
        std::tie(m_properties, m_throw_exception) = this->GetParam();
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
    }
};

TEST_P(SetPropertyTest, SetPropertyTestCase) {
    if (m_properties.size() == 0) {
        OV_ASSERT_NO_THROW(m_plugin->set_property(m_properties));
        return;
    }

    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->set_property(m_properties));
    } else {
        OV_ASSERT_NO_THROW(m_plugin->set_property(m_properties));
    }
}

const std::vector<set_property_params> plugin_set_property_params_test = {
    set_property_params{{{ov::auto_batch_timeout(static_cast<uint32_t>(200))}}, false},
    set_property_params{{{ov::device::priorities("CPU(4)")}}, false},
    set_property_params{{{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("CPU(4)")}}, false},
    set_property_params{{{"XYZ", "200"}}, true},
    set_property_params{{{"XYZ", "200"}, {ov::device::priorities("CPU(4)")}}, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         SetPropertyTest,
                         ::testing::ValuesIn(plugin_set_property_params_test),
                         SetPropertyTest::getTestCaseName);