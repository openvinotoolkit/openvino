// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_common.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"

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

using remote_context_param = std::tuple<ov::AnyMap,  // Set Property
                                        bool>;

class AutoBatchRemoteContextTest : public ::testing::TestWithParam<remote_context_param> {
public:
    ov::AnyMap m_properties;
    bool m_throw_exception;
    std::shared_ptr<NiceMock<MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;
    ov::RemoteContext m_remote_context;

public:
    static std::string getTestCaseName(testing::TestParamInfo<remote_context_param> obj) {
        ov::AnyMap properties;
        bool throw_exception;

        std::tie(properties, throw_exception) = obj.param;
        std::string res = "";
        if (properties.size() > 0) {
            res += "AutoBatchRemoteContext_";
            for (auto& it : properties) {
                res += it.first + "_" + it.second.as<std::string>() + "_";
            }
        }
        if (throw_exception)
            res += "_throw";
        return res;
    }

    void TearDown() override {
        m_core.reset();
        m_plugin.reset();
    }

    void SetUp() override {
        std::tie(m_properties, m_throw_exception) = this->GetParam();
        m_core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        m_plugin->set_core(m_core);

        ON_CALL(*m_core, create_context).WillByDefault(Return(m_remote_context));
        ON_CALL(*m_core, get_default_context).WillByDefault(Return(m_remote_context));
    }
};

TEST_P(AutoBatchRemoteContextTest, CreateContextTestCase) {
    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->create_context(m_properties));
    } else {
        ASSERT_NO_THROW(m_plugin->create_context(m_properties));
    }
}

TEST_P(AutoBatchRemoteContextTest, GetDefaultContextTestCase) {
    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->get_default_context(m_properties));
    } else {
        ASSERT_NO_THROW(m_plugin->get_default_context(m_properties));
    }
}

const std::vector<remote_context_param> remote_context_params_testextTest = {
    remote_context_param{{{}}, true},
    remote_context_param{{{"AUTO_BATCH_TIMEOUT", "200"}}, true},
    remote_context_param{{{"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}}, false},
    remote_context_param{{{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}}, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         AutoBatchRemoteContextTest,
                         ::testing::ValuesIn(remote_context_params_testextTest),
                         AutoBatchRemoteContextTest::getTestCaseName);