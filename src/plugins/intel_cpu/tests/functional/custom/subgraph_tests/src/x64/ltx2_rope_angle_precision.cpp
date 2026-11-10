// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov::test {

// LTX-2 fractional RoPE (diffusers LTX2AudioVideoRotaryPosEmbed, split layout), applied to a projection:
//   coords / max_positions -> (grid * 2 - 1) * freqs -> transpose(-1, -2) + flatten -> Sin/Cos
// with freqs = theta ^ linspace(0, 1, N) * pi / 2 reaching ~1.6e4. The reference model computes this chain in f32 also
// in fp16 / bf16; lowered to f16 / bf16 the angles are off by many radians. KeepRopeAnglesInF32ForModels(LTX2) keeps
// the chain in f32 while the MatMuls run in the requested precision.
class LTX2RopeAnglePrecisionCPUTest : public testing::WithParamInterface<ov::element::Type>, public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& obj) {
        return "inferPRC=" + obj.param.to_string();
    }

protected:
    void SetUp() override {
        const auto infer_prc = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), infer_prc});
        // low precision angles give O(1) errors; honest f16 / bf16 rounding of the MatMuls stays well under this
        rel_threshold = 0.05;
        abs_threshold = 0.5;

        init_input_shapes({InputShape{{1, TOKENS, AXES}, {{1, TOKENS, AXES}}},
                           InputShape{{1, TOKENS, HIDDEN}, {{1, TOKENS, HIDDEN}}}});
        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }

        auto grid = std::make_shared<ov::op::v1::Divide>(
            params[0],
            utils::make_constant(ov::element::f32, ov::Shape{1, 1, AXES}, std::vector<float>{20.0f, 2048.0f, 2048.0f}));
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            grid,
            utils::make_constant(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}));
        auto scaled = std::make_shared<ov::op::v1::Multiply>(
            unsqueeze,
            utils::make_constant(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{2.0f}));
        auto centered = std::make_shared<ov::op::v1::Add>(
            scaled,
            utils::make_constant(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{-1.0f}));
        std::vector<float> freqs(BANDS);
        for (size_t i = 0; i < BANDS; ++i) {
            const auto exponent = static_cast<float>(i) / static_cast<float>(BANDS - 1);
            freqs[i] = std::pow(10000.0f, exponent) * 3.14159265358979323846f / 2.0f;
        }
        auto angles =
            std::make_shared<ov::op::v1::Multiply>(centered,
                                                   utils::make_constant(ov::element::f32, ov::Shape{BANDS}, freqs));
        auto transpose = std::make_shared<ov::op::v1::Transpose>(
            angles,
            utils::make_constant(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 3, 2}));
        auto flat = std::make_shared<ov::op::v1::Reshape>(
            transpose,
            utils::make_constant(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 0, -1}),
            true);
        auto cos = std::make_shared<ov::op::v0::Cos>(flat);
        auto sin = std::make_shared<ov::op::v0::Sin>(flat);

        ov::test::utils::InputGenerateData weights_data(-1, 2, 256);
        auto q = std::make_shared<ov::op::v0::MatMul>(
            params[1],
            utils::make_constant(ov::element::f32, ov::Shape{HIDDEN, HIDDEN}, weights_data));
        auto rotated = std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(q, cos),
                                                         std::make_shared<ov::op::v1::Multiply>(q, sin));
        auto out = std::make_shared<ov::op::v0::MatMul>(
            rotated,
            utils::make_constant(ov::element::f32, ov::Shape{HIDDEN, HIDDEN}, weights_data));
        function = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(out)},
                                               params,
                                               "LTX2RopeAnglePrecision");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();

        // (frame, y, x) positions of a small latent video, as LTX-2 prepares them
        ov::Tensor coords{ov::element::f32, targetInputStaticShapes[0]};
        auto* data = coords.data<float>();
        for (size_t t = 0; t < TOKENS; ++t) {
            data[t * AXES + 0] = static_cast<float>(t / 16) * 8.0f / 24.0f;
            data[t * AXES + 1] = static_cast<float>((t / 4) % 4) * 32.0f;
            data[t * AXES + 2] = static_cast<float>(t % 4) * 32.0f;
        }
        inputs.insert({func_inputs[0].get_node_shared_ptr(), coords});

        utils::InputGenerateData in_data(-1, 2, 256);
        inputs.insert({func_inputs[1].get_node_shared_ptr(),
                       utils::create_and_fill_tensor(ov::element::f32, targetInputStaticShapes[1], in_data)});
    }

    static constexpr size_t TOKENS = 64;
    static constexpr size_t AXES = 3;
    static constexpr size_t BANDS = 16;
    static constexpr size_t HIDDEN = AXES * BANDS;
};

TEST_P(LTX2RopeAnglePrecisionCPUTest, CompareWithRefs) {
    const auto infer_prc = GetParam();
    if (infer_prc == ov::element::bf16 && !(ov::with_cpu_x86_avx512_core() || with_cpu_x86_avx2_vnni_2())) {
        GTEST_SKIP() << "No BF16 support";
    }
    if (infer_prc == ov::element::f16 && !(ov::with_cpu_x86_avx512_core_fp16() || with_cpu_x86_avx2_vnni_2())) {
        GTEST_SKIP() << "No FP16 support";
    }
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_LTX2RopeAnglePrecision,
                         LTX2RopeAnglePrecisionCPUTest,
                         testing::Values(ov::element::f16, ov::element::bf16),
                         LTX2RopeAnglePrecisionCPUTest::getTestCaseName);

}  // namespace ov::test
