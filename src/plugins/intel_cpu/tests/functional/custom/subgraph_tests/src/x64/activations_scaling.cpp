// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// T5-like block: RMSNorm -> MatMul -> Gelu -> MatMul -> Add(residual) -> RMSNorm,
// the second MatMul output exceeds the f16 range
class ActivationsScalingCPUTest : public testing::WithParamInterface<float>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<float>& obj) {
        std::ostringstream result;
        result << "scale_factor=" << obj.param;
        return result.str();
    }

protected:
    static constexpr size_t hidden_size = 64;
    static constexpr size_t inter_size = 256;

    static ov::Output<ov::Node> make_rms_norm(const ov::Output<ov::Node>& x) {
        auto square =
            std::make_shared<ov::op::v1::Power>(x, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.f}));
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(square, axes, true);
        auto eps = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1e-6f});
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(mean, eps));
        auto div = std::make_shared<ov::op::v1::Divide>(x, sqrt);
        auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{hidden_size}, {1.f});
        return std::make_shared<ov::op::v1::Multiply>(div, gamma);
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const float scale_factor = GetParam();
        configuration = {{ov::hint::inference_precision.name(), ov::element::f16}};
        if (scale_factor > 0.f) {
            configuration.insert({ov::hint::activations_scale_factor.name(), scale_factor});
        }
        abs_threshold = 0.05;
        rel_threshold = 0.05;

        init_input_shapes({{{-1, -1, hidden_size}, {{1, 8, hidden_size}}}});

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto norm0 = make_rms_norm(input);
        auto weights0 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{hidden_size, inter_size}, {1.f});
        auto matmul0 = std::make_shared<ov::op::v0::MatMul>(norm0, weights0);
        auto gelu = std::make_shared<ov::op::v7::Gelu>(matmul0);
        auto weights1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{inter_size, hidden_size}, {8.f});
        auto matmul1 = std::make_shared<ov::op::v0::MatMul>(gelu, weights1);
        auto residual = std::make_shared<ov::op::v1::Add>(input, matmul1);
        auto norm1 = make_rms_norm(residual);
        auto result = std::make_shared<ov::op::v0::Result>(norm1);
        function =
            std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "ActivationsScaling");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& param = function->inputs()[0];
        // positive values, so the MatMul outputs do not cancel out
        auto tensor = ov::test::utils::create_and_fill_tensor(param.get_element_type(),
                                                              targetInputStaticShapes[0],
                                                              ov::test::utils::InputGenerateData(1, 10, 1, 1));
        inputs.insert({param.get_node_shared_ptr(), tensor});
    }

    bool output_is_finite() {
        const auto output = inferRequest.get_output_tensor(0);
        const auto* data = output.data<float>();
        return std::all_of(data, data + output.get_size(), [](float v) {
            return std::isfinite(v);
        });
    }
};

namespace {
TEST_P(ActivationsScalingCPUTest, CompareWithRefs) {
    if (!ov::with_cpu_x86_avx512_core_fp16()) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    if (GetParam() > 0.f) {
        run();
        CheckNumberOfNodesWithType(compiledModel, "RMSNorm", 2);
    } else {
        // the block overflows without scaling
        compile_model();
        generate_inputs(targetStaticShapes.front());
        infer();
        ASSERT_FALSE(output_is_finite());
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_ActivationsScaling,
                         ActivationsScalingCPUTest,
                         ::testing::Values(-1.f, 8.f),
                         ActivationsScalingCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
