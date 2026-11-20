// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/util/common_util.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

struct QuantizationParams {
    std::vector<std::vector<float>> inputIntervals;   // quantize intervals before convolution
    std::vector<std::vector<float>> outputIntervals;  // quantize intervals after convolution
    std::vector<size_t> fqConstShapes;                // fq constant shapes
    element::Type expectedPrecision;                  // convolution expected precision
    bool perChannelWeightsScale;                      // use per-channel scale on weights
    double absoluteThreshold = 1e-2;
    std::vector<float> biasOnlyValues{};
};

typedef std::tuple<InputShape,          // input shape
                   element::Type,       // input precision
                   QuantizationParams,  // quantization parameters
                   bool,                // bias presence
                   bool,                // swish presence
                   std::string          // device name
                   >
    ConvAndFQTestParams;

class ConvAndFQ : public testing::WithParamInterface<ConvAndFQTestParams>,
                  virtual public SubgraphBaseTest,
                  public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvAndFQTestParams>& obj) {
        const auto& [inputShape, inputPrecision, quantizationParams, withBias, withSwish, targetName] = obj.param;
        std::ostringstream results;

        results << "IS=" << inputShape << "_InPRC=" << inputPrecision
                << "_ExpectedPRC=" << quantizationParams.expectedPrecision << "_InputIntervals=";
        for (const auto& vecInt : quantizationParams.inputIntervals) {
            results << ov::util::vector_to_string(vecInt) << ",";
        }
        results << "_OutputIntervals=";
        for (const auto& vecInt : quantizationParams.outputIntervals) {
            results << ov::util::vector_to_string(vecInt) << ",";
        }
        results << "_fqShapes=" << ov::util::vector_to_string(quantizationParams.fqConstShapes)
                << "_withBias=" << withBias << "_withSwish=" << withSwish
                << "_perChannelWeightsScale=" << quantizationParams.perChannelWeightsScale
                << "_targetDevice=" << targetName;

        return results.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShape, inputPrecision, quantizationParams, withBias, withSwish, targetName] =
            this->GetParam();
        abs_threshold = quantizationParams.absoluteThreshold;
        if (!quantizationParams.biasOnlyValues.empty()) {
            rel_threshold = 0.0;
        }
        targetDevice = targetName;
        std::tie(inFmts, outFmts, priority, selectedType) = CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type};
        init_input_shapes({inputShape});
        ov::ParameterVector input_params{
            std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputDynamicShapes[0])};

        const auto& inputIntervals = quantizationParams.inputIntervals;
        const auto& outputIntervals = quantizationParams.outputIntervals;
        const auto& fqConstShapes = quantizationParams.fqConstShapes;

        auto fq_before = ov::test::utils::make_fake_quantize(input_params[0],
                                                             inputPrecision,
                                                             256,
                                                             fqConstShapes,
                                                             inputIntervals[0],
                                                             inputIntervals[1],
                                                             inputIntervals[2],
                                                             inputIntervals[3]);

        std::shared_ptr<Node> weights;
        if (quantizationParams.biasOnlyValues.empty()) {
            weights = utils::make_constant(element::i8, {4, 3, 2, 2});
        } else {
            weights =
                op::v0::Constant::create(element::i8, {4, 4, 1, 1}, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        }
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);

        std::shared_ptr<ov::Node> multiply;
        if (quantizationParams.perChannelWeightsScale) {
            const auto scales = quantizationParams.biasOnlyValues.empty()
                                    ? std::vector<float>{0.02F, 0.025F, 0.03F, 0.035F}
                                    : std::vector<float>{0.012F, 0.011F, 0.014F, 0.017F};
            multiply = std::make_shared<op::v1::Multiply>(convert,
                                                          op::v0::Constant::create(element::f32, {4, 1, 1, 1}, scales));
        } else {
            multiply =
                std::make_shared<op::v1::Multiply>(convert,
                                                   op::v0::Constant::create(element::f32, {1, 1, 1, 1}, {0.625f}));
        }

        std::shared_ptr<Node> conv;
        {
            const std::vector<size_t> kernelSize = {1, 1};
            const std::vector<size_t> strides = {1, 1};
            const std::vector<ptrdiff_t> padBegin = {0, 0};
            const std::vector<ptrdiff_t> padEnd = {0, 0};
            const std::vector<size_t> dilation = {1, 1};
            const size_t numOutChannels = 4;
            const op::PadType paddingType = op::PadType::EXPLICIT;
            conv = ov::test::utils::make_convolution(fq_before,
                                                     multiply,
                                                     inputPrecision,
                                                     kernelSize,
                                                     strides,
                                                     padBegin,
                                                     padEnd,
                                                     dilation,
                                                     paddingType,
                                                     numOutChannels);
        }

        auto fqInput = conv;
        if (withBias) {
            const Shape biasShape{1, 4, 1, 1};
            std::shared_ptr<Node> bias;
            if (quantizationParams.biasOnlyValues.empty()) {
                bias = std::make_shared<op::v0::Convert>(utils::make_constant(element::f16, biasShape), element::f32);
            } else {
                bias = op::v0::Constant::create(element::f32, biasShape, quantizationParams.biasOnlyValues);
            }
            fqInput = std::make_shared<op::v1::Add>(conv, bias);
        }
        if (withSwish) {
            fqInput = ov::test::utils::make_activation(fqInput,
                                                       inputPrecision,
                                                       ov::test::utils::ActivationTypes::Swish,
                                                       {},
                                                       {1.0F});
        }

        auto fq_after = ov::test::utils::make_fake_quantize(fqInput,
                                                            inputPrecision,
                                                            256,
                                                            {},
                                                            {outputIntervals[0][0]},
                                                            {outputIntervals[1][0]},
                                                            {outputIntervals[2][0]},
                                                            {outputIntervals[3][0]});

        if (!quantizationParams.biasOnlyValues.empty()) {
            function = create_ov_model(inputPrecision, input_params, fq_after, "ConvFQ");
            return;
        }

        auto matmul_const = ov::test::utils::make_constant(ov::element::i8, {1, 1});
        auto convert_mm = std::make_shared<op::v0::Convert>(matmul_const, inputPrecision);
        auto multiply_mm =
            std::make_shared<op::v1::Multiply>(convert_mm, op::v0::Constant::create(inputPrecision, {1, 1}, {0.1}));
        const auto matMul = std::make_shared<ov::op::v0::MatMul>(fq_after, multiply_mm, false, false);

        function = create_ov_model(inputPrecision, input_params, matMul, "ConvFQ");
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const auto& funcInput = funcInputs[0];
        if (!std::get<QuantizationParams>(GetParam()).biasOnlyValues.empty()) {
            ov::Tensor tensor(funcInput.get_element_type(), targetInputStaticShapes[0]);
            std::fill_n(tensor.data<float>(), tensor.get_size(), 0.0F);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
            return;
        }
        ov::Tensor tensor;
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 2;
        in_data.resolution = 256;
        tensor =
            ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[0], in_data);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
    void checkConvolutionPrecision(ov::element::Type expectedPrecision) {
        const auto convolutions = GetNodesWithTypes(compiledModel.get_runtime_model(), {"Convolution"});
        ASSERT_EQ(convolutions.size(), 1u);
        EXPECT_EQ(convolutions.front()->get_rt_info().at(ov::exec_model_info::RUNTIME_PRECISION).as<element::Type>(),
                  expectedPrecision);
    }
};

TEST_P(ConvAndFQ, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();

    const auto& [inputShape, inputPrecision, quantizationParams, withBias, withSwish, targetName] = this->GetParam();
    checkConvolutionPrecision(quantizationParams.expectedPrecision);
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

namespace {

std::vector<InputShape> inputShapes{{{}, {{4, 3, 2, 2}}}, {{-1, 3, -1, 2}, {{1, 3, 4, 2}}}};

#if defined(OPENVINO_ARCH_ARM64)
const element::Type expectedConvPrecBySignedFQRange = element::i8;
const element::Type expectedConvPrecByUnsignedFQRange = element::u8;
#else
const element::Type expectedConvPrecBySignedFQRange = element::f32;
const element::Type expectedConvPrecByUnsignedFQRange = element::f32;
#endif

const std::vector<std::vector<float>> signedIntervals{{-1.28f}, {1.27f}, {-1.28f}, {1.27f}};
const std::vector<std::vector<float>> unsignedIntervals{{0.f}, {2.55f}, {0.f}, {2.55f}};

#if defined(OPENVINO_ARCH_ARM64)
const QuantizationParams zeroPointRoundingParams{{{0.F}, {0.64194745F}, {0.F}, {0.64194745F}},
                                                 {{-0.78389084F}, {0.777767F}, {-0.78389084F}, {0.777767F}},
                                                 {},
                                                 element::u8,
                                                 true,
                                                 0.001,
                                                 {0.012F, -0.018F, 0.032F, -0.041F}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvAndFQZeroPointRounding_CPU,
                         ConvAndFQ,
                         ::testing::Combine(::testing::Values(InputShape{{}, {{1, 4, 4, 4}}}),
                                            ::testing::Values(element::f32),
                                            ::testing::Values(zeroPointRoundingParams),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvAndFQ::getTestCaseName);
#endif

std::vector<QuantizationParams> quantizationParams{
    {signedIntervals, signedIntervals, {}, expectedConvPrecBySignedFQRange, false},        // i8 -> i8, per-tensor
    {unsignedIntervals, unsignedIntervals, {}, expectedConvPrecByUnsignedFQRange, false},  // u8 -> u8, per-tensor
    {signedIntervals, signedIntervals, {}, expectedConvPrecBySignedFQRange, true},         // i8 -> i8, per-channel
};

std::vector<QuantizationParams> mixedOutputQuantizationParams{
    {unsignedIntervals, signedIntervals, {}, expectedConvPrecByUnsignedFQRange, false},  // u8 -> i8
    {signedIntervals, unsignedIntervals, {}, expectedConvPrecBySignedFQRange, false},    // i8 -> u8
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvAndFQ_CPU,
                         ConvAndFQ,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(quantizationParams),
                                            ::testing::Values(false, true),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvAndFQ::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvAndFQMixedOutput_CPU,
                         ConvAndFQ,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(mixedOutputQuantizationParams),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvAndFQ::getTestCaseName);

// upstream ACL's dequantize-to-F32 convolution path (int8_dequantize_f32_path) only accepts
// QASYMM8_SIGNED (i8) src, not u8, so a u8-src Swish conv currently falls back to plain f32
// need to update this test later after upstream ACL has supported u8 src
std::vector<QuantizationParams> withSwishQuantizationParams{
    {signedIntervals, signedIntervals, {}, expectedConvPrecBySignedFQRange, false},  // i8 -> i8, per-tensor
    {unsignedIntervals, unsignedIntervals, {}, element::f32, false},  // u8 -> u8, per-tensor (fallbacks to f32)
    {signedIntervals, signedIntervals, {}, expectedConvPrecBySignedFQRange, true},  // i8 -> i8, per-channel
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvAndFQ_withSwish_CPU,
                         ConvAndFQ,
                         ::testing::Combine(::testing::Values(inputShapes.front()),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(withSwishQuantizationParams),
                                            ::testing::Values(true),  // withBias
                                            ::testing::Values(true),  // withSwish
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvAndFQ::getTestCaseName);

std::vector<QuantizationParams> withSwishDynamicQuantizationParams{
    {signedIntervals, signedIntervals, {}, expectedConvPrecBySignedFQRange, false},  // i8 -> i8, per-tensor
    {unsignedIntervals, unsignedIntervals, {}, element::f32, false},  // u8 -> u8, per-tensor (fallbacks to f32)
    {signedIntervals, signedIntervals, {}, element::f32, true},       // i8 -> f32, per-channel (fallbacks to f32)
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvAndFQ_withSwish_CPU_Dynamic,
                         ConvAndFQ,
                         ::testing::Combine(::testing::Values(inputShapes.back()),
                                            ::testing::Values(element::f32),
                                            ::testing::ValuesIn(withSwishDynamicQuantizationParams),
                                            ::testing::Values(true),  // withBias
                                            ::testing::Values(true),  // withSwish
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConvAndFQ::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
