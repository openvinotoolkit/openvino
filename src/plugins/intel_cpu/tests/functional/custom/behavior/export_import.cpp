// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-corer: Apache-2.0
//

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include <openvino/opsets/opset9.hpp>

namespace {

using PropertiesParams = std::tuple<std::string, std::vector<ov::AnyMap>>;

class ExportOptimalNumStreams : public ::testing::TestWithParam<PropertiesParams> {};

std::shared_ptr<ov::Model> MakeMatMulModel() {
    const ov::Shape input_shape = {1, 4096};

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(input_shape))};
    auto matmul_const = ov::test::utils::make_constant(ov::element::f32, {4096, 1024});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], matmul_const);

    auto add_const = ov::test::utils::make_constant(ov::element::f32, {1, 1024});
    auto add = ov::test::utils::make_eltwise(matmul, add_const, ov::test::utils::EltwiseTypes::ADD);
    auto softmax = std::make_shared<ov::opset9::Softmax>(add);

    ov::NodeVector results{softmax};
    return std::make_shared<ov::Model>(results, params, "MatMulModel");
}

TEST_P(ExportOptimalNumStreams, OptimalNumStreams) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto original_model = MakeMatMulModel();
    ov::Core core;
    std::string device_name;
    std::vector<ov::AnyMap> properties;
    std::tie(device_name, properties) = GetParam();
    auto original_properties_input = properties[0];
    auto new_properties_input = properties[1];

    auto GetProperties = [&](ov::CompiledModel& network) {
        std::vector<std::string> properties;
        properties.push_back(network.get_property(ov::hint::performance_mode.name()).as<std::string>());
        properties.push_back(network.get_property(ov::num_streams.name()).as<std::string>());
        properties.push_back(network.get_property(ov::hint::scheduling_core_type.name()).as<std::string>());
        properties.push_back(network.get_property(ov::hint::enable_hyper_threading.name()).as<std::string>());
        properties.push_back(network.get_property(ov::hint::enable_cpu_pinning.name()).as<std::string>());
        properties.push_back(network.get_property(ov::inference_num_threads.name()).as<std::string>());
        properties.push_back(network.get_property(ov::hint::num_requests.name()).as<std::string>());
        return properties;
    };

    auto original_network = core.compile_model(original_model, device_name, original_properties_input);
    auto original_properties_output = GetProperties(original_network);

    auto new_network = core.compile_model(original_model, device_name, new_properties_input);
    auto new_properties_output = GetProperties(new_network);

    std::stringstream exported_model;
    original_network.export_model(exported_model);

    // import_model with original config can create the same multi_thread setting as compile_model
    {
        std::stringstream ss(exported_model.str());
        auto imported_network = core.import_model(ss, device_name, original_properties_input);
        auto imported_properties_output = GetProperties(imported_network);

        EXPECT_EQ(original_properties_output[0], imported_properties_output[0]);
        EXPECT_EQ(original_properties_output[1], imported_properties_output[1]);
        EXPECT_EQ(original_properties_output[2], imported_properties_output[2]);
        EXPECT_EQ(original_properties_output[3], imported_properties_output[3]);
        EXPECT_EQ(original_properties_output[4], imported_properties_output[4]);
        EXPECT_EQ(original_properties_output[5], imported_properties_output[5]);
        EXPECT_EQ(original_properties_output[6], imported_properties_output[6]);
    }

    // import_model with new properties can create the same multi_thread setting as compile_model with new properties
    {
        std::stringstream ss(exported_model.str());
        auto imported_network = core.import_model(ss, device_name, new_properties_input);
        auto imported_properties_output = GetProperties(imported_network);

        EXPECT_EQ(new_properties_output[0], imported_properties_output[0]);
        EXPECT_EQ(new_properties_output[1], imported_properties_output[1]);
        EXPECT_EQ(new_properties_output[2], imported_properties_output[2]);
        EXPECT_EQ(new_properties_output[3], imported_properties_output[3]);
        EXPECT_EQ(new_properties_output[4], imported_properties_output[4]);
        EXPECT_EQ(new_properties_output[5], imported_properties_output[5]);
        EXPECT_EQ(new_properties_output[6], imported_properties_output[6]);
    }
}

const std::vector<ov::AnyMap> testing_property_for_streams = {{ov::num_streams(1)}, {ov::num_streams(2)}};

const std::vector<ov::AnyMap> testing_property_for_threads = {{ov::inference_num_threads(1)},
                                                              {ov::inference_num_threads(4)}};

const std::vector<ov::AnyMap> testing_property_for_performance_mode = {
    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};

const std::vector<ov::AnyMap> testing_property_for_scheduling_core_type = {
    {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ANY_CORE)},
    {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY)},
    {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY)}};

const std::vector<ov::AnyMap> testing_property_for_enable_hyper_threading = {{ov::hint::enable_hyper_threading(true)},
                                                                             {ov::hint::enable_hyper_threading(false)}};

const std::vector<ov::AnyMap> testing_property_for_enable_cpu_pinning = {{ov::hint::enable_cpu_pinning(true)},
                                                                         {ov::hint::enable_cpu_pinning(false)}};

INSTANTIATE_TEST_SUITE_P(smoke_ExportImportTest,
                        ExportOptimalNumStreams,
                        ::testing::Combine(::testing::Values(std::string("CPU")),
                                           ::testing::Values(testing_property_for_streams,
                                                             testing_property_for_threads,
                                                             testing_property_for_performance_mode,
                                                             testing_property_for_scheduling_core_type,
                                                             testing_property_for_enable_hyper_threading,
                                                             testing_property_for_enable_cpu_pinning)));

}  // namespace
