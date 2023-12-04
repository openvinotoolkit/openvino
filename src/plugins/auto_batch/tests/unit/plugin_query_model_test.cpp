// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_common.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

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

using query_model_params = std::tuple<ov::AnyMap,  // Set Property
                                      bool>;

class QueryModelTest : public ::testing::TestWithParam<query_model_params> {
public:
    ov::AnyMap m_properties;
    bool m_throw_exception;
    std::shared_ptr<NiceMock<ov::MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;
    std::shared_ptr<ov::Model> m_model;
    ov::SupportedOpsMap m_supported_ops_map;

public:
    static std::string getTestCaseName(testing::TestParamInfo<query_model_params> obj) {
        ov::AnyMap properties;
        bool throw_exception;

        std::tie(properties, throw_exception) = obj.param;
        std::string res = "";
        if (properties.size() > 0) {
            res += "QueryModel_";
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
        m_model.reset();
    }

    void SetUp() override {
        std::tie(m_properties, m_throw_exception) = this->GetParam();
        m_model = ngraph::builder::subgraph::makeMultiSingleConv();
        m_core = std::shared_ptr<NiceMock<ov::MockICore>>(new NiceMock<ov::MockICore>());
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        m_plugin->set_core(m_core);

        ON_CALL(*m_core, query_model).WillByDefault(Return(m_supported_ops_map));
    }
};

TEST_P(QueryModelTest, QueryModelTestCase) {
    if (m_throw_exception) {
        ASSERT_ANY_THROW(m_plugin->query_model(m_model, m_properties));
    } else {
        ASSERT_NO_THROW(m_plugin->query_model(m_model, m_properties));
    }
}

const std::vector<query_model_params> query_model_params_test = {
    query_model_params{{{}}, true},
    query_model_params{{{"AUTO_BATCH_TIMEOUT", "200"}}, true},
    query_model_params{{{"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}}, false},
    query_model_params{{{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(4)"}}, false},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         QueryModelTest,
                         ::testing::ValuesIn(query_model_params_test),
                         QueryModelTest::getTestCaseName);
