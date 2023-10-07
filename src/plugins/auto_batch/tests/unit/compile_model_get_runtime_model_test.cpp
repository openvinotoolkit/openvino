// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_common.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;

using namespace ov::mock_autobatch_plugin;

class CompileModelGetRuntimeModelTest : public ::testing::Test {
public:
    std::shared_ptr<NiceMock<ov::MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;
    std::shared_ptr<ov::Model> m_model;

    ov::SoPtr<MockICompiledModel> m_mock_compile_model;
    std::shared_ptr<MockICompiledModel> m_mock_i_compile_model;
    std::shared_ptr<ov::ICompiledModel> m_auto_batch_compile_model;

    std::shared_ptr<NiceMock<MockIPlugin>> m_hardware_plugin;

public:
    void TearDown() override {
        m_core.reset();
        m_plugin.reset();
        m_model.reset();
        m_mock_i_compile_model.reset();
        m_mock_compile_model = {};
        m_auto_batch_compile_model.reset();
    }

    void SetUp() override {
        m_model = ngraph::builder::subgraph::makeMultiSingleConv();
        m_core = std::shared_ptr<NiceMock<ov::MockICore>>(new NiceMock<ov::MockICore>());
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        m_plugin->set_core(m_core);
        m_hardware_plugin = std::shared_ptr<NiceMock<MockIPlugin>>(new NiceMock<MockIPlugin>());
        m_mock_i_compile_model = std::make_shared<NiceMock<MockICompiledModel>>(m_model, m_hardware_plugin);
        m_mock_compile_model = {m_mock_i_compile_model, {}};

        ON_CALL(*m_core,
                compile_model(MatcherCast<const std::shared_ptr<const ov::Model>&>(_),
                              MatcherCast<const std::string&>(_),
                              _))
            .WillByDefault(Return(m_mock_compile_model));

        ON_CALL(*m_core,
                compile_model(MatcherCast<const std::shared_ptr<const ov::Model>&>(_),
                              MatcherCast<const ov::SoPtr<ov::IRemoteContext>&>(_),
                              _))
            .WillByDefault(Return(m_mock_compile_model));

        ON_CALL(*m_core, get_property(_, StrEq("PERFORMANCE_HINT")))
            .WillByDefault(Return(ov::hint::PerformanceMode::THROUGHPUT));

        ON_CALL(*m_core, get_property(_, StrEq("OPTIMAL_BATCH_SIZE"), _))
            .WillByDefault(Return(static_cast<unsigned int>(16)));

        ON_CALL(*m_core, get_property(_, StrEq("PERFORMANCE_HINT_NUM_REQUESTS")))
            .WillByDefault(Return(static_cast<uint32_t>(12)));

        ON_CALL(*m_core, get_property(_, StrEq("GPU_MEMORY_STATISTICS"), _))
            .WillByDefault([](const std::string& device, const std::string& key, const ov::AnyMap& options) {
                std::map<std::string, uint64_t> ret = {{"xyz", 1024}};
                return ret;
            });

        ON_CALL(*m_core, get_property(_, StrEq("GPU_DEVICE_TOTAL_MEM_SIZE"), _)).WillByDefault(Return("10240"));

        ON_CALL(*m_mock_i_compile_model.get(), get_runtime_model()).WillByDefault(Return(m_model));

        const ov::AnyMap configs = {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(16)"}};

        ASSERT_NO_THROW(m_auto_batch_compile_model = m_plugin->compile_model(m_model, configs));
    }
};

TEST_F(CompileModelGetRuntimeModelTest, CompileModelGetRuntimeModelTestCase) {
    ASSERT_NO_THROW(m_auto_batch_compile_model->get_runtime_model());
}
