// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
// Use bias
typedef std::tuple<bool> Conv3DNormActivationsScalingParams;

// Output head of a causal video VAE decoder (LTX-2): RMS norm over channels -> SiLU -> causal temporal padding
// (first frame repeated) -> Conv3D. With activations scaling, f16 inference keeps a scale operation next to the
// convolution that gets fused into it as a oneDNN binary post-op. The post-op operand must be described with the
// rank of the 5D convolution output, otherwise oneDNN rejects the primitive and no implementation can be selected.
class Conv3DNormActivationsScaling : public testing::WithParamInterface<Conv3DNormActivationsScalingParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Conv3DNormActivationsScalingParams>& params) {
        const auto& [with_bias] = params.param;
        std::ostringstream result;
        result << "bias=" << with_bias;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        configuration.insert({ov::hint::activations_scale_factor.name(), 8.f});
        abs_threshold = 5e-2;
        rel_threshold = 2e-2;
        const auto& [with_bias] = GetParam();
        const size_t in_ch = 32, out_ch = 48;
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.f, 0.05f);

        auto x = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{1, in_ch, 4, 16, 16});
        auto two = ov::op::v0::Constant::create(type, ov::Shape{}, {2.f});
        auto eps = ov::op::v0::Constant::create(type, ov::Shape{}, {1e-6f});
        auto channel_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto mean_sq = std::make_shared<ov::op::v1::ReduceMean>(std::make_shared<ov::op::v1::Power>(x, two), channel_axis, true);
        auto rms = std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(mean_sq, eps));
        auto act = std::make_shared<ov::op::v4::Swish>(std::make_shared<ov::op::v1::Divide>(x, rms));

        auto i64_1d = [](int64_t v) {
            return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {v});
        };
        auto first_frame = std::make_shared<ov::op::v8::Slice>(act, i64_1d(0), i64_1d(1), i64_1d(1), i64_1d(2));
        auto padded = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_frame, first_frame, act}, 2);

        std::vector<float> w(out_ch * in_ch * 27);
        for (auto& v : w)
            v = dist(gen);
        auto weights = ov::op::v0::Constant::create(type, ov::Shape{out_ch, in_ch, 3, 3, 3}, w);
        std::shared_ptr<ov::Node> out = std::make_shared<ov::op::v1::Convolution>(padded,
                                                                                  weights,
                                                                                  ov::Strides{1, 1, 1},
                                                                                  ov::CoordinateDiff{0, 1, 1},
                                                                                  ov::CoordinateDiff{0, 1, 1},
                                                                                  ov::Strides{1, 1, 1});
        if (with_bias) {
            std::vector<float> b(out_ch);
            for (auto& v : b)
                v = dist(gen);
            out = std::make_shared<ov::op::v1::Add>(out, ov::op::v0::Constant::create(type, ov::Shape{1, out_ch, 1, 1, 1}, b));
        }
        function = std::make_shared<ov::Model>(ov::OutputVector{out}, ov::ParameterVector{x});
    }

    const ov::element::Type type = ov::element::f32;
};

TEST_P(Conv3DNormActivationsScaling, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_GPU_Conv3DNormActivationsScaling,
                         Conv3DNormActivationsScaling,
                         ::testing::Values(Conv3DNormActivationsScalingParams{false}, Conv3DNormActivationsScalingParams{true}),
                         Conv3DNormActivationsScaling::getTestCaseName);

}  // namespace
