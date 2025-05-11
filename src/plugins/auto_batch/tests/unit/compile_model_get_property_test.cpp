// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_common.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"
#include "common_test_utils/subgraph_builders/multi_single_conv.hpp"

using get_property_param = std::tuple<std::string,  // Property need to be set
                                      bool>;        // Throw exception

class CompileModelGetPropertyTest : public ::testing::TestWithParam<get_property_param> {
public:
    std::string m_properity_name;
    bool m_throw_exception;
    std::shared_ptr<NiceMock<ov::MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;
    std::shared_ptr<ov::Model> m_model;

    ov::SoPtr<MockICompiledModel> m_mock_compile_model;
    std::shared_ptr<MockICompiledModel> m_mock_i_compile_model;
    std::shared_ptr<NiceMock<MockIPlugin>> m_hardware_plugin;

    std::shared_ptr<ov::ICompiledModel> auto_batch_compile_model;

public:
    static std::string getTestCaseName(testing::TestParamInfo<get_property_param> obj) {
        std::string properity_name;
        bool throw_exception;
        std::tie(properity_name, throw_exception) = obj.param;

        std::string res;
        res += "_" + properity_name;
        if (throw_exception)
            res += "throw";

        return res;
    }

    void TearDown() override {
        m_core.reset();
        m_plugin.reset();
        m_model.reset();
        m_mock_i_compile_model.reset();
        m_mock_compile_model = {};
        auto_batch_compile_model.reset();
    }

    void SetUp() override {
        std::tie(m_properity_name, m_throw_exception) = this->GetParam();
        m_model = ov::test::utils::make_multi_single_conv();
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

        const ov::AnyMap configs = {{ov::auto_batch_timeout(static_cast<uint32_t>(200))}, {ov::device::priorities("CPU(16)")}};
        OV_ASSERT_NO_THROW(auto_batch_compile_model = m_plugin->compile_model(m_model, configs));

        std::string network_name = m_model.get()->get_name();
        std::vector<ov::PropertyName> supported_props = {ov::optimal_batch_size, ov::cache_dir};

        ON_CALL(*m_mock_i_compile_model.get(), get_property(StrEq(ov::supported_properties.name())))
            .WillByDefault(Return(ov::Any(supported_props)));

        ON_CALL(*m_mock_i_compile_model.get(), get_property(StrEq("PERFORMANCE_HINT_NUM_REQUESTS")))
            .WillByDefault(Return("0"));

        ON_CALL(*m_mock_i_compile_model.get(), get_property(StrEq("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
            .WillByDefault(Return("12"));

        ON_CALL(*m_mock_i_compile_model.get(), get_property(StrEq("NETWORK_NAME")))
            .WillByDefault(Return(network_name.c_str()));

        ON_CALL(*m_mock_i_compile_model.get(), get_property(StrEq("EXECUTION_DEVICES"))).WillByDefault(Return("CPU"));

        ON_CALL(*m_mock_i_compile_model.get(), get_property(StrEq("CACHE_DIR"))).WillByDefault(Return("./abc"));

        ON_CALL(*m_mock_i_compile_model.get(), get_property(StrEq("OPTIMAL_BATCH_SIZE"))).WillByDefault(Return("16"));
    }
};

TEST_P(CompileModelGetPropertyTest, CompileModelGetPropertyTestCase) {
    if (m_throw_exception)
        ASSERT_ANY_THROW(auto_batch_compile_model->get_property(m_properity_name));
    else
        OV_ASSERT_NO_THROW(auto_batch_compile_model->get_property(m_properity_name));
}

const std::vector<get_property_param> compile_model_get_property_param_test = {
    get_property_param{ov::optimal_number_of_infer_requests.name(), false},
    get_property_param{ov::model_name.name(), false},
    get_property_param{ov::supported_properties.name(), false},
    get_property_param{ov::execution_devices.name(), false},
    get_property_param{ov::device::priorities.name(), false},
    get_property_param{ov::auto_batch_timeout.name(), false},
    get_property_param{ov::cache_dir.name(), false},
    // Config in dependent m_plugin
    get_property_param{ov::optimal_batch_size.name(), false},
    // Incorrect Property
    get_property_param{"INCORRECT_METRIC", true},
    get_property_param{"INCORRECT_CONFIG", true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         CompileModelGetPropertyTest,
                         ::testing::ValuesIn(compile_model_get_property_param_test),
                         CompileModelGetPropertyTest::getTestCaseName);
