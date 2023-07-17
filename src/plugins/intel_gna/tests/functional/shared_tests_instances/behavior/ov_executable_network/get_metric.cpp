// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gna/gna_config.hpp>

#include "behavior/compiled_model/properties.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"

using namespace ov::test::behavior;

namespace {
//
// Executable Network GetMetric
//
class OVClassNetworkTestGNA : public ::testing::Test {
public:
    std::shared_ptr<ngraph::Function> gnaSimpleNetwork;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        auto param0 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape(1, 1024));
        auto reshape = std::make_shared<ngraph::opset8::Reshape>(
            param0,
            std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                                                       ngraph::Shape{4},
                                                       ngraph::Shape{1, 1, 1, 1024}),
            false);
        param0->set_friendly_name("input");
        auto conv1 = ngraph::builder::makeConvolution(reshape,
                                                      ngraph::element::Type_t::f32,
                                                      {1, 7},
                                                      {1, 1},
                                                      {0, 0},
                                                      {0, 0},
                                                      {1, 1},
                                                      ngraph::op::PadType::EXPLICIT,
                                                      4);
        auto result = std::make_shared<ngraph::opset8::Result>(conv1);
        gnaSimpleNetwork =
            std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0});
        gnaSimpleNetwork->set_friendly_name("GnaSingleConv");
    }
};

class OVClassBaseTestGNAP : public OVClassNetworkTestGNA, public ::testing::WithParamInterface<std::string> {
public:
    std::string deviceName;

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVClassNetworkTestGNA::SetUp();
        deviceName = GetParam();
    }
};

class OVClassCompiledModelGetPropertyTestForSpecificConfigGNA
    : public OVClassNetworkTestGNA,
      public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, ov::Any>>> {
protected:
    std::string deviceName;
    std::string configKey;
    ov::Any configValue;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVClassNetworkTestGNA::SetUp();
        deviceName = std::get<0>(GetParam());
        std::tie(configKey, configValue) = std::get<1>(GetParam());
    }
};

using OVGNAClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS = OVClassBaseTestGNAP;
using OVGNAClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS = OVClassBaseTestGNAP;
using OVGNAClassExecutableNetworkGetMetricTest_NETWORK_NAME = OVClassBaseTestGNAP;
using OVGNAClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS = OVClassBaseTestGNAP;
using OVGNAClassExecutableNetworkGetMetricTest_ThrowsUnsupported = OVClassBaseTestGNAP;

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVGNAClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
                         ::testing::Values("GNA" /*, "MULTI:GNA"*/, "HETERO:GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVGNAClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
                         ::testing::Values("GNA" /*, "MULTI:GNA"*/, "HETERO:GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVGNAClassExecutableNetworkGetMetricTest_NETWORK_NAME,
                         ::testing::Values("GNA" /*, "MULTI:GNA"*/, "HETERO:GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVGNAClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
                         ::testing::Values("GNA" /*, "MULTI:GNA"*/, "HETERO:GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVGNAClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
                         ::testing::Values("GNA" /*, "MULTI:GNA"*/, "HETERO:GNA"));

//
// Executable Network GetConfig / SetConfig
//

using OVGNAClassExecutableNetworkGetConfigTest = OVClassBaseTestGNAP;
using OVGNAClassExecutableNetworkSetConfigTest = OVClassBaseTestGNAP;

INSTANTIATE_TEST_SUITE_P(moke_OVClassCompiledModelGetConfigTest,
                         OVGNAClassExecutableNetworkGetConfigTest,
                         ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetIncorrectPropertyTest,
                         OVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetConfigTest,
                         OVClassCompiledModelGetConfigTest,
                         ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelSetIncorrectConfigTest,
                         OVGNAClassExecutableNetworkSetConfigTest,
                         ::testing::Values("GNA"));

using OVGNAClassExecutableNetworkSupportedConfigTest = OVClassCompiledModelGetPropertyTestForSpecificConfigGNA;
using OVGNAClassExecutableNetworkUnsupportedConfigTest = OVClassCompiledModelGetPropertyTestForSpecificConfigGNA;

IE_SUPPRESS_DEPRECATED_START
INSTANTIATE_TEST_SUITE_P(
    smoke_OVClassExecutableNetworkSupportedConfigTest,
    OVGNAClassExecutableNetworkSupportedConfigTest,
    ::testing::Combine(
        ::testing::Values("GNA"),
        ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_HW),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_EXACT),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_AUTO))));
IE_SUPPRESS_DEPRECATED_END

INSTANTIATE_TEST_SUITE_P(
    smoke_OVClassExecutableNetworkUnsupportedConfigTest,
    OVGNAClassExecutableNetworkUnsupportedConfigTest,
    ::testing::Combine(::testing::Values("GNA"),
                       ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE),
                                                        InferenceEngine::GNAConfigParams::GNA_SW_FP32),
                                         std::make_pair(GNA_CONFIG_KEY(SCALE_FACTOR), "5"),
                                         std::make_pair(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)),
                                         std::make_pair(GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)))));

using OVClassExecutableNetworkSetConfigFromFp32Test = OVClassCompiledModelGetPropertyTestForSpecificConfigGNA;
using OVClassExecutableNetworkSetConfigROProperties = OVClassCompiledModelGetPropertyTestForSpecificConfigGNA;

TEST_P(OVClassExecutableNetworkSetConfigFromFp32Test, SetConfigFromFp32Throws) {
    ov::Core ie;

    ov::AnyMap initialConfig;

    ov::CompiledModel exeNetwork =
        ie.compile_model(gnaSimpleNetwork,
                         deviceName,
                         ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32));

    ASSERT_THROW(exeNetwork.set_property({{configKey, configValue}}), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_OVClassExecutableNetworkSetConfigFromFp32Test,
    OVClassExecutableNetworkSetConfigFromFp32Test,
    ::testing::Combine(
        ::testing::Values("GNA"),
        ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_HW),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE),
                                         InferenceEngine::GNAConfigParams::GNA_HW_WITH_SW_FBACK),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_EXACT),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_FP32),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_AUTO),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::HW),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::HW_WITH_SW_FBACK),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::AUTO))));

TEST_P(OVClassExecutableNetworkSetConfigROProperties, SetConfigROPropertiesThrows) {
    ov::Core ie;
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel exeNetwork = ie.compile_model(gnaSimpleNetwork, deviceName);

    ASSERT_NO_THROW(properties = exeNetwork.get_property(ov::supported_properties));

    auto it = find(properties.begin(), properties.end(), configKey);
    ASSERT_TRUE(it != properties.end());
    ASSERT_FALSE(it->is_mutable());

    ASSERT_THROW(exeNetwork.set_property({{configKey, configValue}}), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_OVClassExecutableNetworkSetConfigROProperties,
    OVClassExecutableNetworkSetConfigROProperties,
    ::testing::Combine(
        ::testing::Values("GNA"),
        ::testing::Values(ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"0", 1.0f}}),
                          ov::hint::inference_precision(ngraph::element::i8),
                          ov::hint::num_requests(2),
                          ov::intel_gna::pwl_design_algorithm(ov::intel_gna::PWLDesignAlgorithm::UNIFORM_DISTRIBUTION),
                          ov::intel_gna::pwl_max_error_percent(0.2),
                          ov::intel_gna::firmware_model_image_path(""),
                          ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::GNA_3_0),
                          ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_3_0))));

using OVClassExecutableNetworkDevicePropertiesTest = OVClassCompiledModelGetPropertyTestForSpecificConfigGNA;
TEST_P(OVClassExecutableNetworkDevicePropertiesTest, DevicePropertiesNoThrow) {
    ov::Core ie;
    ASSERT_NO_THROW(auto compiled_model =
                        ie.compile_model(gnaSimpleNetwork,
                                         deviceName,
                                         ov::device::properties("GNA", ov::AnyMap{{configKey, configValue}})));
}

INSTANTIATE_TEST_SUITE_P(
    smoke_OVClassExecutableNetworkDevicePropertiesTest,
    OVClassExecutableNetworkDevicePropertiesTest,
    ::testing::Combine(
        ::testing::Values("HETERO:GNA"),
        ::testing::Values(ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::HW),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::HW_WITH_SW_FBACK),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
                          ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::AUTO),
                          ov::intel_gna::scale_factors_per_input(std::map<std::string, float>{{"input", 1.0f}}),
                          ov::hint::inference_precision(ov::element::i8),
                          ov::hint::inference_precision(ov::element::i16),
                          ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                          ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                          ov::hint::num_requests(1),
                          ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_2_0),
                          ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::GNA_3_0),
                          ov::intel_gna::execution_target(ov::intel_gna::HWGeneration::UNDEFINED),
                          ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::GNA_2_0),
                          ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::GNA_3_0),
                          ov::intel_gna::compile_target(ov::intel_gna::HWGeneration::UNDEFINED),
                          ov::intel_gna::pwl_design_algorithm(ov::intel_gna::PWLDesignAlgorithm::RECURSIVE_DESCENT),
                          ov::intel_gna::pwl_design_algorithm(ov::intel_gna::PWLDesignAlgorithm::UNIFORM_DISTRIBUTION),
                          ov::intel_gna::pwl_max_error_percent(0.05),
                          ov::log::level(ov::log::Level::NO))));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(smoke_OVCompiledModelIncorrectDevice,
                         OVCompiledModelIncorrectDevice,
                         ::testing::Values("GNA"));
}  // namespace
