// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using LPConvFallbackParams = std::tuple<ov::AnyMap, CPUSpecificParams>;

class LPConvFallback : public testing::WithParamInterface<LPConvFallbackParams>,
                       virtual public SubgraphBaseStaticTest,
                       public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LPConvFallbackParams>& obj) {
        const auto& [additionalConfig, cpuParams] = obj.param;
        std::ostringstream result;
        result << "IS=(1.16.32.32)_Config=";
        if (additionalConfig.empty()) {
            result << "Default";
        } else {
            result << additionalConfig.at(ov::hint::inference_precision.name()).as<std::string>();
        }
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [additionalConfig, cpuParams] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration = additionalConfig;
        configuration.insert(ov::num_streams(1));
        configuration.insert(ov::inference_num_threads(1));
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        // The shared CPU test configuration supplies f32 when no hint is specified.
        expectedPrecision = additionalConfig.empty()
                                ? element::f32
                                : additionalConfig.at(ov::hint::inference_precision.name()).as<element::Type>();
        if (expectedPrecision == element::f16) {
            const auto capabilities = core->get_property(targetDevice, ov::device::capabilities);
            if (std::find(capabilities.begin(), capabilities.end(), "FP16") == capabilities.end()) {
                GTEST_SKIP() << "CPU does not support FP16 inference";
            }
        }
        selectedType = makeSelectedTypeStr(".*acl", expectedPrecision);

        ov::ParameterVector params{std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 16, 32, 32})};
        const auto inputFQ =
            utils::make_fake_quantize(params[0], element::f32, 256, {}, {-0.5f}, {127.0f}, {-0.5f}, {127.0f});
        std::vector<int8_t> weightsData(32 * 16 * 3 * 3);
        for (size_t index = 0; index < weightsData.size(); ++index) {
            weightsData[index] = static_cast<int8_t>(static_cast<int>((index * 73) % 255) - 127);
        }
        const auto weights = op::v0::Constant::create(element::i8, Shape{32, 16, 3, 3}, weightsData);
        const auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        ov::disable_constant_folding(convert);
        std::vector<float> scales(32);
        for (size_t index = 0; index < scales.size(); ++index) {
            scales[index] = 0.0009765625f * static_cast<float>(1 + index % 4);
        }
        const auto scale = op::v0::Constant::create(element::f32, Shape{32, 1, 1, 1}, scales);
        const auto multiply = std::make_shared<op::v1::Multiply>(convert, scale);
        const auto convolution = std::make_shared<op::v1::Convolution>(inputFQ,
                                                                       multiply,
                                                                       Strides{2, 2},
                                                                       CoordinateDiff{1, 1},
                                                                       CoordinateDiff{1, 1},
                                                                       Strides{1, 1});
        // A channel-dependent bias keeps Add from folding into the scalar FQ ranges.
        std::vector<float> biasData(32);
        for (size_t index = 0; index < biasData.size(); ++index) {
            biasData[index] = 0.375f + static_cast<float>(index) * 0.03125f;
        }
        const auto bias = op::v0::Constant::create(element::f32, Shape{1, 32, 1, 1}, biasData);
        const auto add = std::make_shared<op::v1::Add>(convolution, bias);
        const auto outputFQ =
            utils::make_fake_quantize(add, element::f32, 256, {}, {-0.5f}, {127.0f}, {-0.5f}, {127.0f});
        function = std::make_shared<Model>(OutputVector{outputFQ}, params, "LPConvFallback");
        // Allow two output FQ quanta (0.5 each) for f16 rounding after moving scales.
        abs_threshold = 1.0f + 1e-5f;
        rel_threshold = 0;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& parameter = function->get_parameters().front();
        utils::InputGenerateData inputData;
        inputData.start_from = -0.5f;
        inputData.range = 130;
        inputData.resolution = 4;
        const auto tensor = utils::create_and_fill_tensor(element::f32, targetInputStaticShapes[0], inputData);
        inputs.insert({parameter, tensor});
    }

    element::Type expectedPrecision;
};

TEST_P(LPConvFallback, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Convolution", 1);
    CheckPluginRelatedResults(compiledModel, "Convolution");
    const auto convolutions = GetNodesWithTypes(compiledModel.get_runtime_model(), {"Convolution"});
    for (const auto& node : convolutions) {
        EXPECT_EQ(node->get_output_element_type(0), expectedPrecision);
        for (const auto& input : node->inputs()) {
            EXPECT_EQ(input.get_element_type(), expectedPrecision);
        }
    }
}

namespace {

const std::vector<CPUSpecificParams> cpuParams{
    {{nhwc}, {nhwc}, {}, "acl"},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPConvFallback_CPU,
    LPConvFallback,
    ::testing::Combine(::testing::Values(cpu_f16_plugin_config, cpu_f32_plugin_config, empty_plugin_config),
                       ::testing::ValuesIn(cpuParams)),
    LPConvFallback::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
