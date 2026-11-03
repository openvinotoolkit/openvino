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
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {
// Temporal downsampler of a causal video VAE encoder (LTX-2): the residual path is a space-to-depth reshape followed by
// a mean over channel groups of size 1, i.e. a ReduceMean whose reduced axis has size 1 (keep_dims=false). Next to an
// RMS norm elsewhere in the model, the reduction is assigned to oneDNN, which rejects identity reductions, so no
// implementation could be selected. Such reductions must fall back to the OCL kernel.
class ReduceIdentityAxes : public testing::WithParamInterface<ov::element::Type>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& params) {
        return "inferPRC=" + params.param.to_string();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration.insert({ov::hint::inference_precision.name(), GetParam()});
        abs_threshold = 5e-2;
        rel_threshold = 2e-2;
        const int64_t C = 64, H = 16, W = 24;
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.f, 0.01f);
        auto i64 = [](std::vector<int64_t> v) {
            return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
        };
        auto random = [&](ov::Shape shape) {
            std::vector<float> v(ov::shape_size(shape));
            for (auto& x : v)
                x = dist(gen);
            return ov::op::v0::Constant::create(ov::element::f32, shape, v);
        };
        const auto c = static_cast<size_t>(C), h = static_cast<size_t>(H), w = static_cast<size_t>(W);

        // RMS norm branch
        auto z = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, c, 1, h, w});
        auto mean_sq =
            std::make_shared<ov::op::v1::ReduceMean>(std::make_shared<ov::op::v1::Power>(z, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.f})),
                                                     i64({1}),
                                                     true);
        auto rms = std::make_shared<ov::op::v0::Sqrt>(
            std::make_shared<ov::op::v1::Add>(mean_sq, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1e-6f})));
        auto norm = std::make_shared<ov::op::v1::Divide>(z, rms);

        // downsampler branch: causal time padding, Conv3D, temporal space-to-depth on both paths
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, c, 1, h, w});
        auto first = std::make_shared<ov::op::v8::Slice>(x, i64({0}), i64({1}), i64({1}), i64({2}));
        auto xt = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first, x}, 2);
        auto pad = std::make_shared<ov::op::v0::Tile>(std::make_shared<ov::op::v8::Slice>(xt, i64({0}), i64({1}), i64({1}), i64({2})), i64({1, 1, 2, 1, 1}));
        auto conv =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Convolution>(std::make_shared<ov::op::v0::Concat>(ov::OutputVector{pad, xt}, 2),
                                                                                        random({c, c, 3, 3, 3}),
                                                                                        ov::Strides{1, 1, 1},
                                                                                        ov::CoordinateDiff{0, 1, 1},
                                                                                        ov::CoordinateDiff{0, 1, 1},
                                                                                        ov::Strides{1, 1, 1}),
                                              random({1, c, 1, 1, 1}));
        auto space_to_depth = [&](const ov::Output<ov::Node>& v) {
            auto r = std::make_shared<ov::op::v1::Reshape>(v, i64({1, C, -1, 2, H, 1, W, 1}), false);
            auto t = std::make_shared<ov::op::v1::Transpose>(r, i64({0, 1, 3, 5, 7, 2, 4, 6}));
            return std::make_shared<ov::op::v1::Reshape>(t, i64({1, -1, 1, H, W}), false);
        };
        auto groups = std::make_shared<ov::op::v1::Reshape>(space_to_depth(xt), i64({1, -1, 1, 1, H, W}), false);
        auto residual = std::make_shared<ov::op::v1::ReduceMean>(groups, i64({2}), false);  // reduced axis has size 1
        auto down = std::make_shared<ov::op::v1::Add>(space_to_depth(conv), residual);

        function = std::make_shared<ov::Model>(ov::OutputVector{down, norm}, ov::ParameterVector{x, z});
    }
};

TEST_P(ReduceIdentityAxes, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_GPU_ReduceIdentityAxes,
                         ReduceIdentityAxes,
                         ::testing::Values(ov::element::f16, ov::element::f32),
                         ReduceIdentityAxes::getTestCaseName);

}  // namespace
