// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/*
 * FakeQuantize -> MatMul(transpose_b) with i8 weights and a per-output-channel dequantization scale, f32 output:
 *   Parameter -> FakeQuantize -> MatMul <- Multiply(Convert(i8 weights), scales[N, 1])
 * With an activation zero point of 0 (u8) or 128 (shifted to i8) no zero-point correction is needed, so on ARM the
 * dequantization Multiply is fused into FullyConnected as per-channel dqScales. With a dynamic input shape the
 * ACL low-precision executor rejects per-channel scales on update and the oneDNN FullyConnected executor takes
 * the node; it must apply attrs.dqScales, as the oneDNN MatMul and Convolution executors do. Without them the
 * output was the unscaled integer accumulator.
 */

typedef std::tuple<InputShape,              // input shape
                   std::pair<float, float>  // FakeQuantize input range (low, high)
                   >
    FCPerChannelDequantizationParams;

class FCPerChannelDequantization : public testing::WithParamInterface<FCPerChannelDequantizationParams>,
                                   virtual public SubgraphBaseTest,
                                   public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FCPerChannelDequantizationParams>& obj) {
        const auto& [inputShape, range] = obj.param;
        std::ostringstream results;
        results << "IS=" << inputShape << "_FQ=" << range.first << "_" << range.second;
        return results.str();
    }

protected:
    static constexpr size_t inChannels = 64;
    static constexpr size_t outChannels = 32;

    void SetUp() override {
        const auto& [inputShape, range] = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        // f32 keeps the expected values exact; with f16 the faulty output overflows to inf instead.
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f32});
        abs_threshold = 2e-2f;
        fqRange = range;
        init_input_shapes({inputShape});

        ov::ParameterVector input_params{std::make_shared<ov::op::v0::Parameter>(element::f32, inputDynamicShapes[0])};
        auto fq = ov::test::utils::make_fake_quantize(input_params[0],
                                                      element::f32,
                                                      256,
                                                      {},
                                                      {range.first},
                                                      {range.second},
                                                      {range.first},
                                                      {range.second});

        // Distinct scales per output channel, so applying channel 0's scale everywhere is detectable.
        std::vector<float> scales(outChannels);
        for (size_t i = 0; i < outChannels; ++i) {
            scales[i] = 0.002f * static_cast<float>(i + 1);
        }
        auto weights = ov::test::utils::make_constant(element::i8, ov::Shape{outChannels, inChannels});
        auto convert_weights = std::make_shared<op::v0::Convert>(weights, element::f32);
        auto dequantized_weights =
            std::make_shared<op::v1::Multiply>(convert_weights,
                                               op::v0::Constant::create(element::f32, {outChannels, 1}, scales));
        auto fc = std::make_shared<ov::op::v0::MatMul>(fq, dequantized_weights, false, true);

        function = create_ov_model(element::f32, input_params, fc, "FCPerChannelDequantization");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto funcInputs = function->inputs();
        const auto& funcInput = funcInputs[0];
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = fqRange.first;
        in_data.range = static_cast<uint32_t>(std::ceil(fqRange.second - fqRange.first));
        in_data.resolution = 256;
        auto tensor =
            ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[0], in_data);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }

    std::pair<float, float> fqRange;
};

TEST_P(FCPerChannelDequantization, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

namespace {

const std::vector<InputShape> inputShapes{
    {{}, {{2, 64}}},
    {{-1, 64}, {{2, 64}, {5, 64}}},
    {{-1, -1, 64}, {{1, 4, 64}, {2, 3, 64}}},
};

// Zero point 0 (u8) and 128 (shifted to i8) fuse the scales into FullyConnected; zero point 85 keeps a
// separate dequantization and serves as a control.
const std::vector<std::pair<float, float>> fqRanges{
    {0.f, 2.55f},
    {-1.28f, 1.27f},
    {-0.5f, 1.f},
};

INSTANTIATE_TEST_SUITE_P(smoke_FCPerChannelDequantization_CPU,
                         FCPerChannelDequantization,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(fqRanges)),
                         FCPerChannelDequantization::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
