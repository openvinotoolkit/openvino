// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"

/* MatMul followed by two constant Adds, the first one on a compressed bias, as the PyTorch frontend
 * produces for `linear(x, w, b.float()) + c` with an f16/bf16 `b`. The frontend marks the bias Convert
 * as decompression and disables its constant folding, so the two Adds are not merged before
 * FullyConnectedBiasFusion and both get fused into the FC bias, one after the other.
 *
 *     Param   Constant(W)
 *        \     /
 *        MatMul      Constant(b, f16/bf16)
 *           \          |
 *            \       Convert(f32)  <- decompression, no constant folding
 *             \      /
 *               Add      Constant(c, f32)
 *                 \      /
 *                   Add
 *                    |
 *                  Result
 */

namespace ov::test {

using FullyConnectedTwoBiasAddsParams = std::tuple<ov::element::Type,  // bias precision
                                                   size_t>;            // output channels

class FullyConnectedTwoBiasAddsTest : public testing::WithParamInterface<FullyConnectedTwoBiasAddsParams>,
                                      virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FullyConnectedTwoBiasAddsParams>& obj) {
        const auto& [bias_type, channels] = obj.param;
        return "biasType=" + bias_type.get_type_name() + "_OC=" + std::to_string(channels);
    }

protected:
    void SetUp() override {
        const auto& [bias_type, channels] = GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        configuration.insert(ov::hint::inference_precision(ov::element::f32));

        constexpr size_t ic = 64;
        const ov::Shape input_shape{2, ic};

        std::vector<float> weights(channels * ic);
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 64.f;
        }
        // Bias values are exact in f16 and bf16 and far from zero, so dropping either Add is visible.
        std::vector<float> bias(channels), shift(channels);
        for (size_t i = 0; i < channels; i++) {
            bias[i] = 1.f + static_cast<float>(i % 8) / 8.f;
            shift[i] = -3.f + static_cast<float>(i % 5) / 4.f;
        }

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{channels, ic}, weights);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights_const, false, true);

        auto bias_const = ov::op::v0::Constant::create(bias_type, ov::Shape{channels}, bias);
        auto bias_convert = std::make_shared<ov::op::v0::Convert>(bias_const, ov::element::f32);
        ov::mark_as_decompression(bias_convert);
        ov::disable_constant_folding(bias_const);
        ov::disable_constant_folding(bias_convert);
        auto add_bias = std::make_shared<ov::op::v1::Add>(matmul, bias_convert);

        auto shift_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{channels}, shift);
        auto add_shift = std::make_shared<ov::op::v1::Add>(add_bias, shift_const);

        function = std::make_shared<ov::Model>(ov::OutputVector{add_shift},
                                               ov::ParameterVector{param},
                                               "FullyConnectedTwoBiasAdds");
    }
};

TEST_P(FullyConnectedTwoBiasAddsTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_FullyConnectedTwoBiasAdds,
                         FullyConnectedTwoBiasAddsTest,
                         ::testing::Combine(::testing::Values(ov::element::f16, ov::element::bf16),
                                            ::testing::Values(16, 2048)),
                         FullyConnectedTwoBiasAddsTest::getTestCaseName);

}  // namespace ov::test
