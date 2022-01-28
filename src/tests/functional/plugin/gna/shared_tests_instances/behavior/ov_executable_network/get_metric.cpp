// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/get_metric.hpp"

#include "openvino/runtime/intel_gna/properties.hpp"
#include <gna/gna_config.hpp>

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
        auto reshape = std::make_shared<ngraph::opset8::Reshape>(param0,
                std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{4}, ngraph::Shape{1, 1, 1, 1024}), false);
        auto conv1 = ngraph::builder::makeConvolution(reshape, ngraph::element::Type_t::f32, {1, 7}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                ngraph::op::PadType::EXPLICIT, 4);
        auto result = std::make_shared<ngraph::opset8::Result>(conv1);
        gnaSimpleNetwork = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param0});
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

class OVClassExecutableNetworkGetMetricTestForSpecificConfigGNA :
        public OVClassNetworkTestGNA,
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

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GNA" /*, "MULTI:GNA", "HETERO:GNA" */));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GNA" /*, "MULTI:GNA",  "HETERO:GNA" */));

// TODO: this metric is not supported by the plugin
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GNA", "MULTI:GNA", "HETERO:GNA"));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("GNA" /*, "MULTI:GNA", "HETERO:GNA" */));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetMetricTest,
        OVClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::Values("GNA", /* "MULTI:GNA", */ "HETERO:GNA"));

//
// Executable Network GetConfig / SetConfig
//

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkGetConfigTest,
        OVClassExecutableNetworkGetConfigTest,
        ::testing::Values("GNA"));

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassExecutableNetworkSetConfigTest,
        OVClassExecutableNetworkSetConfigTest,
        ::testing::Values("GNA"));

IE_SUPPRESS_DEPRECATED_START
// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_OVClassExecutableNetworkSupportedConfigTest,
        OVClassExecutableNetworkSupportedConfigTest,
        ::testing::Combine(
        ::testing::Values("GNA"),
        ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_HW),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_SW_EXACT),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_AUTO))));
IE_SUPPRESS_DEPRECATED_END

// TODO: Convolution with 3D input is not supported on GNA
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_OVClassExecutableNetworkUnsupportedConfigTest,
        OVClassExecutableNetworkUnsupportedConfigTest,
        ::testing::Combine(::testing::Values("GNA"),
                           ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE),
                                                            InferenceEngine::GNAConfigParams::GNA_SW_FP32),
                                             std::make_pair(GNA_CONFIG_KEY(SCALE_FACTOR), "5"),
                                             std::make_pair(CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES)),
                                             std::make_pair(GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)))));

using OVClassExecutableNetworkSetConfigFromFp32Test = OVClassExecutableNetworkGetMetricTestForSpecificConfigGNA;
using OVClassExecutableNetworkSetConfigROProperties = OVClassExecutableNetworkGetMetricTestForSpecificConfigGNA;

TEST_P(OVClassExecutableNetworkSetConfigFromFp32Test, SetConfigFromFp32Throws) {
    ov::Core ie;

    ov::AnyMap initialConfig;

    ov::CompiledModel exeNetwork = ie.compile_model(gnaSimpleNetwork, deviceName,
        ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32));

    ASSERT_THROW(exeNetwork.set_property({{configKey, configValue}}), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassExecutableNetworkSetConfigFromFp32Test,
        OVClassExecutableNetworkSetConfigFromFp32Test,
        ::testing::Combine(
        ::testing::Values("GNA"),
        ::testing::Values(std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_HW),
                          std::make_pair(GNA_CONFIG_KEY(DEVICE_MODE), InferenceEngine::GNAConfigParams::GNA_HW_WITH_SW_FBACK),
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

//
// Hetero Executable Network GetMetric
//

// TODO: verify hetero interop
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassHeteroExecutableNetworlGetMetricTest,
        OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GNA"));

// TODO: verify hetero interop
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassHeteroExecutableNetworlGetMetricTest,
        OVClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GNA"));

// TODO: verify hetero interop
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_OVClassHeteroExecutableNetworlGetMetricTest,
        OVClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("GNA"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroExecutableNetworlGetMetricTest,
        OVClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values("GNA"));
}  // namespace
