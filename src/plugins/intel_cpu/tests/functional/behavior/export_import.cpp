// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-corer: Apache-2.0
//

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"
#include "common_test_utils/test_common.hpp"
#include "ngraph_functions/builders.hpp"


#include <openvino/opsets/opset9.hpp>
#include <ie/ie_core.hpp>

namespace {

using PropertiesParams = std::tuple<std::string, std::vector<ov::AnyMap>>;

class ExportOptimalNumStreams : public ::testing::TestWithParam<PropertiesParams> {};

typedef std::tuple<std::pair<std::string, ov::Any>, std::pair<std::string, ov::Any>> OnePropertyParam;

typedef std::tuple<std::string,
                   OnePropertyParam,
                   OnePropertyParam,
                   OnePropertyParam,
                   OnePropertyParam,
                   OnePropertyParam,
                   OnePropertyParam,
                   OnePropertyParam>
    MultiPropertiesParams;

class MultiExportOptimalNumStreams : public ::testing::TestWithParam<MultiPropertiesParams> {};

std::shared_ptr<ov::Model> MakeMatMulModel() {
    const ov::Shape input_shape = {1, 4096};
    const ov::element::Type precision = ov::element::f32;

    auto params = ngraph::builder::makeParams(precision, {input_shape});
    auto matmul_const = ngraph::builder::makeConstant(precision, {4096, 1024}, std::vector<float>{}, true);
    auto matmul = ngraph::builder::makeMatMul(params[0], matmul_const);

    auto add_const = ngraph::builder::makeConstant(precision, {1, 1024}, std::vector<float>{}, true);
    auto add = ngraph::builder::makeEltwise(matmul, add_const, ngraph::helpers::EltwiseTypes::ADD);
    auto softmax = std::make_shared<ov::opset9::Softmax>(add);

    ngraph::NodeVector results{softmax};
    return std::make_shared<ov::Model>(results, params, "MatMulModel");
}

void CheckOptimalNumStreams(std::string& device_name, ov::AnyMap& original_properties_input, ov::AnyMap& new_properties_input) {
    auto original_model = MakeMatMulModel();
    ov::Core core;

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

    // import_model with no config can create the same multi_thread setting as compile_model
    {
        std::stringstream ss(exported_model.str());
        auto imported_network = core.import_model(ss, device_name);
        auto imported_properties_output = GetProperties(imported_network);

        EXPECT_EQ(original_properties_output[0], imported_properties_output[0]);
        EXPECT_EQ(original_properties_output[1], imported_properties_output[1]);
        EXPECT_EQ(original_properties_output[2], imported_properties_output[2]);
        EXPECT_EQ(original_properties_output[3], imported_properties_output[3]);
        EXPECT_EQ(original_properties_output[4], imported_properties_output[4]);
        EXPECT_EQ(original_properties_output[5], imported_properties_output[5]);
        EXPECT_EQ(original_properties_output[6], imported_properties_output[6]);
    }

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

TEST_P(ExportOptimalNumStreams, OptimalNumStreams) {
    std::string device_name;
    std::vector<ov::AnyMap> properties;
    std::tie(device_name, properties) = GetParam();
    auto original_properties_input = properties[0];
    auto new_properties_input = properties[1];

    CheckOptimalNumStreams(device_name, original_properties_input, new_properties_input);
}

const std::vector<std::vector<ov::AnyMap>> testing_properties = {
    {{ov::num_streams(1)}, {ov::num_streams(2)}},

    {{ov::inference_num_threads(1)}, {ov::inference_num_threads(4)}},

    {{ov::hint::num_requests(1)}, {ov::hint::num_requests(4)}},

    {{ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
     {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},

    {{ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ANY_CORE)},
     {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY)}},

    {{ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY)},
     {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY)}},

    {{ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ANY_CORE)},
     {ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY)}},

    {{ov::hint::enable_hyper_threading(true)}, {ov::hint::enable_hyper_threading(false)}},

    {{ov::hint::enable_cpu_pinning(true)}, {ov::hint::enable_cpu_pinning(false)}}};

TEST_P(MultiExportOptimalNumStreams, MultiOptimalNumStreams) {
    std::string device_name;
    OnePropertyParam performance_param, schedule_core_type_param, hyper_thread_param, cpu_pinning_param, streams_param,
        threads_param, request_param;
    std::tie(device_name,
             performance_param,
             schedule_core_type_param,
             hyper_thread_param,
             cpu_pinning_param,
             streams_param,
             threads_param,
             request_param) = GetParam();
    std::pair<std::string, ov::Any> original_perf, new_perf;
    std::tie(original_perf, new_perf) = performance_param;

    std::pair<std::string, ov::Any> original_schedule_core_type, new_schedule_core_type;
    std::tie(original_schedule_core_type, new_schedule_core_type) = schedule_core_type_param;

    std::pair<std::string, ov::Any> original_hyper_thread, new_hyper_thread;
    std::tie(original_hyper_thread, new_hyper_thread) = hyper_thread_param;

    std::pair<std::string, ov::Any> original_cpu_pinning, new_cpu_pinning;
    std::tie(original_cpu_pinning, new_cpu_pinning) = cpu_pinning_param;

    std::pair<std::string, ov::Any> original_streams, new_streams;
    std::tie(original_streams, new_streams) = streams_param;

    std::pair<std::string, ov::Any> original_threads, new_threads;
    std::tie(original_threads, new_threads) = threads_param;

    std::pair<std::string, ov::Any> original_request, new_request;
    std::tie(original_request, new_request) = request_param;

    // std::cout << "original param: " << std::endl;
    // std::cout << original_perf.first << " - " << original_perf.second.as<std::string>() << std::endl;
    // std::cout << original_schedule_core_type.first << " - " << original_schedule_core_type.second.as<std::string>()
    //           << std::endl;
    // std::cout << original_hyper_thread.first << " - " << original_hyper_thread.second.as<std::string>() << std::endl;
    // std::cout << original_cpu_pinning.first << " - " << original_cpu_pinning.second.as<std::string>() << std::endl;
    // std::cout << original_streams.first << " - " << original_streams.second.as<std::string>() << std::endl;
    // std::cout << original_threads.first << " - " << original_threads.second.as<std::string>() << std::endl;
    // std::cout << original_request.first << " - " << original_request.second.as<std::string>() << std::endl;

    // std::cout << "\nnew param: " << std::endl;
    // std::cout << new_perf.first << " - " << new_perf.second.as<std::string>() << std::endl;
    // std::cout << new_schedule_core_type.first << " - " << new_schedule_core_type.second.as<std::string>() <<
    // std::endl; std::cout << new_hyper_thread.first << " - " << new_hyper_thread.second.as<std::string>() <<
    // std::endl; std::cout << new_cpu_pinning.first << " - " << new_cpu_pinning.second.as<std::string>() << std::endl;
    // std::cout << new_streams.first << " - " << new_streams.second.as<std::string>() << std::endl;
    // std::cout << new_threads.first << " - " << new_threads.second.as<std::string>() << std::endl;
    // std::cout << new_request.first << " - " << new_request.second.as<std::string>() << std::endl;

    ov::AnyMap original_properties, new_properties;
    original_properties.insert(original_perf);
    original_properties.insert(original_schedule_core_type);
    original_properties.insert(original_hyper_thread);
    original_properties.insert(original_cpu_pinning);
    original_properties.insert(original_streams);
    original_properties.insert(original_threads);
    original_properties.insert(original_request);

    new_properties.insert(new_perf);
    new_properties.insert(new_schedule_core_type);
    new_properties.insert(new_hyper_thread);
    new_properties.insert(new_cpu_pinning);
    new_properties.insert(new_streams);
    new_properties.insert(new_threads);
    new_properties.insert(new_request);

    CheckOptimalNumStreams(device_name, original_properties, new_properties);
}

const std::vector<std::pair<std::string, ov::Any>> performance_property = {
    std::make_pair(ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT),
    std::make_pair(ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY)};

const std::vector<std::pair<std::string, ov::Any>> scheduling_core_type_property = {
    std::make_pair(ov::hint::scheduling_core_type.name(), ov::hint::SchedulingCoreType::ANY_CORE),
    std::make_pair(ov::hint::scheduling_core_type.name(), ov::hint::SchedulingCoreType::PCORE_ONLY),
    std::make_pair(ov::hint::scheduling_core_type.name(), ov::hint::SchedulingCoreType::ECORE_ONLY)};

const std::vector<std::pair<std::string, ov::Any>> hyper_threading_property = {
    std::make_pair(ov::hint::enable_hyper_threading.name(), true),
    std::make_pair(ov::hint::enable_hyper_threading.name(), false)};

const std::vector<std::pair<std::string, ov::Any>> cpu_pinning_property = {
    std::make_pair(ov::hint::enable_cpu_pinning.name(), true),
    std::make_pair(ov::hint::enable_cpu_pinning.name(), false)};

const std::vector<std::pair<std::string, ov::Any>> streams_property = {std::make_pair(ov::num_streams.name(), 1),
                                                                       std::make_pair(ov::num_streams.name(), 2)};

const std::vector<std::pair<std::string, ov::Any>> threads_property = {
    std::make_pair(ov::inference_num_threads.name(), 8),
    std::make_pair(ov::inference_num_threads.name(), 24)};

const std::vector<std::pair<std::string, ov::Any>> request_property = {
    std::make_pair(ov::hint::num_requests.name(), 2),
    std::make_pair(ov::hint::num_requests.name(), 4)};

const auto one_properties_performance =
    ::testing::Combine(::testing::ValuesIn(performance_property), ::testing::ValuesIn(performance_property));

const auto one_properties_schedule = ::testing::Combine(::testing::ValuesIn(scheduling_core_type_property),
                                                        ::testing::ValuesIn(scheduling_core_type_property));

const auto one_properties_hyper_thread =
    ::testing::Combine(::testing::ValuesIn(hyper_threading_property), ::testing::ValuesIn(hyper_threading_property));

const auto one_properties_cpu_pinning =
    ::testing::Combine(::testing::ValuesIn(cpu_pinning_property), ::testing::ValuesIn(cpu_pinning_property));

const auto one_properties_streams =
    ::testing::Combine(::testing::ValuesIn(streams_property), ::testing::ValuesIn(streams_property));

const auto one_properties_threads =
    ::testing::Combine(::testing::ValuesIn(threads_property), ::testing::ValuesIn(threads_property));

const auto one_properties_request =
    ::testing::Combine(::testing::ValuesIn(request_property), ::testing::ValuesIn(request_property));

INSTANTIATE_TEST_CASE_P(smoke_ExportImportTest,
                        ExportOptimalNumStreams,
                        ::testing::Combine(::testing::Values(std::string("CPU")),
                                           ::testing::ValuesIn(testing_properties)));

INSTANTIATE_TEST_CASE_P(smoke_ExportImportTest,
                        MultiExportOptimalNumStreams,
                        ::testing::Combine(::testing::Values(std::string("CPU")),
                                           one_properties_performance,
                                           one_properties_schedule,
                                           one_properties_hyper_thread,
                                           one_properties_cpu_pinning,
                                           one_properties_streams,
                                           one_properties_threads,
                                           one_properties_request));

}  // namespace
