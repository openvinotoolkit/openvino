// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// Affine-free RMSNorm followed by a dynamic AdaLN-style scale, as in DiT diffusion transformers:
//   y = x * rsqrt(mean(x^2, -1) + eps) * (1 + scale)
// Must fuse into the internal RMS op so the variance math stays f32 under bf16 enforcement.
class RMSNormNoGammaCPUTest : public SubgraphBaseTest, public testing::WithParamInterface<ov::element::Type> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& obj) {
        return "netPRC=" + obj.param.to_string();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        const auto infer_precision = GetParam();
        configuration.insert({ov::hint::inference_precision.name(), infer_precision});
        rel_threshold = abs_threshold = infer_precision == ov::element::bf16 ? 0.02 : 0.001;

        const std::vector<InputShape> input_shapes = {
            {{-1, -1, HIDDEN_SIZE}, {{1, 8, HIDDEN_SIZE}, {2, 3, HIDDEN_SIZE}}},  // hidden states
            {{-1, 1, HIDDEN_SIZE}, {{1, 1, HIDDEN_SIZE}, {2, 1, HIDDEN_SIZE}}},   // dynamic AdaLN scale
        };
        init_input_shapes(input_shapes);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1])};

        // FCs around the pattern trigger bf16 markup
        auto proj_weights = utils::make_constant(ov::element::f32, ov::Shape{HIDDEN_SIZE, HIDDEN_SIZE});
        auto x = std::make_shared<ov::op::v0::MatMul>(params[0], proj_weights);

        auto pow_const = utils::make_constant(ov::element::f32, ov::Shape{}, std::vector<float>{2.0F});
        auto pow = std::make_shared<ov::op::v1::Power>(x, pow_const);
        auto mean_axes = utils::make_constant(ov::element::i32, ov::Shape{1}, std::vector<int>{-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(pow, mean_axes, true);
        auto eps = utils::make_constant(ov::element::f32, ov::Shape{}, std::vector<float>{1e-6F});
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, eps);
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);
        auto one = utils::make_constant(ov::element::f32, ov::Shape{}, std::vector<float>{1.0F});
        auto rsqrt = std::make_shared<ov::op::v1::Divide>(one, sqrt);
        auto norm = std::make_shared<ov::op::v1::Multiply>(x, rsqrt);

        auto scale = std::make_shared<ov::op::v1::Add>(params[1], one);
        auto modulated = std::make_shared<ov::op::v1::Multiply>(norm, scale);

        auto fc_weights = utils::make_constant(ov::element::f32, ov::Shape{HIDDEN_SIZE, HIDDEN_SIZE});
        auto matmul = std::make_shared<ov::op::v0::MatMul>(modulated, fc_weights);

        function = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(matmul)},
                                               params,
                                               "RMSNormNoGamma");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            utils::InputGenerateData in_data;
            in_data.start_from = -1;
            in_data.range = 2;
            in_data.resolution = 256;
            auto tensor =
                utils::create_and_fill_tensor(funcInputs[i].get_element_type(), targetInputStaticShapes[i], in_data);
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }

    static constexpr size_t HIDDEN_SIZE = 128;
};

TEST_P(RMSNormNoGammaCPUTest, CompareWithRefs) {
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP() << "The RMS kernel requires avx2";
    }
    if (GetParam() == ov::element::bf16 && !ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "No bf16 hardware support";
    }
    run();
    CheckNumberOfNodesWithType(compiledModel, "RMS", 1);
}

INSTANTIATE_TEST_SUITE_P(smoke_RMSNormNoGamma_CPU,
                         RMSNormNoGammaCPUTest,
                         testing::Values(ov::element::f32, ov::element::bf16),
                         RMSNormNoGammaCPUTest::getTestCaseName);

}  // namespace ov::test
