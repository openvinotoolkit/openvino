// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_sin_cos_angles_to_keep_in_mixed_precision.hpp"

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;

namespace {

std::shared_ptr<v0::Constant> f32_const(const ov::Shape& shape, float value) {
    return v0::Constant::create(ov::element::f32, shape, std::vector<float>(ov::shape_size(shape), value));
}

std::shared_ptr<v0::Constant> i64_const(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

// Sin and Cos of the same angles, with every node of the angle path marked when requested
std::shared_ptr<ov::Model> make_tables(const ov::Output<ov::Node>& angles,
                                       const ov::ParameterVector& params,
                                       ov::NodeVector path,
                                       bool marked) {
    auto sin = std::make_shared<v0::Sin>(angles);
    auto cos = std::make_shared<v0::Cos>(angles);
    path.push_back(sin);
    path.push_back(cos);
    if (marked) {
        for (const auto& node : path) {
            ov::disable_conversion(node, ov::element::f16);
        }
    }
    return std::make_shared<ov::Model>(ov::OutputVector{sin, cos}, params);
}

// LTX-Video: grid * freqs + shift -> Transpose -> Reshape
std::shared_ptr<ov::Model> make_shifted_grid_rope(bool marked) {
    auto grid = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 3, 1});
    auto mul = std::make_shared<v1::Multiply>(grid, f32_const({4}, 1570.8f));
    auto add = std::make_shared<v1::Add>(mul, f32_const({}, -1.0f));
    auto transpose = std::make_shared<v1::Transpose>(add, i64_const({0, 1, 3, 2}));
    auto reshape = std::make_shared<v1::Reshape>(transpose, i64_const({1, -1, 12}), false);
    return make_tables(reshape, {grid}, {mul, add, transpose, reshape}, marked);
}

// LTX-2 video: (grid * 2 - 1) * freqs -> Transpose -> Reshape
std::shared_ptr<ov::Model> make_centered_grid_rope(bool marked) {
    auto grid = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 3, 1});
    auto scaled = std::make_shared<v1::Multiply>(grid, f32_const({}, 2.0f));
    auto centered = std::make_shared<v1::Subtract>(scaled, f32_const({}, 1.0f));
    auto mul = std::make_shared<v1::Multiply>(centered, f32_const({4}, 1570.8f));
    auto transpose = std::make_shared<v1::Transpose>(mul, i64_const({0, 1, 3, 2}));
    auto reshape = std::make_shared<v1::Reshape>(transpose, i64_const({1, -1, 12}), false);
    return make_tables(reshape, {grid}, {scaled, centered, mul, transpose, reshape}, marked);
}

// LTX-2 audio and cross-attention: a single position axis, so the transpose is exported as a Reshape
std::shared_ptr<ov::Model> make_single_axis_rope(bool marked) {
    auto grid = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1, 1});
    auto scaled = std::make_shared<v1::Multiply>(grid, f32_const({}, 2.0f));
    auto centered = std::make_shared<v1::Subtract>(scaled, f32_const({}, 1.0f));
    auto mul = std::make_shared<v1::Multiply>(centered, f32_const({4}, 1570.8f));
    auto swap = std::make_shared<v1::Reshape>(mul, i64_const({1, -1, 4, 1}), false);
    auto flatten = std::make_shared<v1::Reshape>(swap, i64_const({1, -1, 4}), false);
    return make_tables(flatten, {grid}, {scaled, centered, mul, swap, flatten}, marked);
}

// LTX-2 text connectors: positions from the sequence length, angles fed to Sin/Cos without any layout op
std::shared_ptr<ov::Model> make_sequence_rope(bool marked) {
    auto hidden = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 64});
    auto shape = std::make_shared<v3::ShapeOf>(hidden);
    auto seq_len = std::make_shared<v8::Gather>(shape, i64_const({1}), i64_const({0}));
    auto squeezed = std::make_shared<v1::Reshape>(seq_len, i64_const({}), false);
    auto range = std::make_shared<v4::Range>(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                             squeezed,
                                             v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                             ov::element::i64);
    auto positions = std::make_shared<v0::Convert>(range, ov::element::f32);
    auto grid = std::make_shared<v1::Divide>(positions, f32_const({}, 4096.0f));
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(grid, i64_const({-1}));
    auto scaled = std::make_shared<v1::Multiply>(unsqueeze, f32_const({}, 2.0f));
    auto centered = std::make_shared<v1::Subtract>(scaled, f32_const({}, 1.0f));
    auto mul = std::make_shared<v1::Multiply>(centered, f32_const({32}, 1570.8f));
    return make_tables(mul,
                       {hidden},
                       {seq_len, squeezed, range, positions, grid, unsqueeze, scaled, centered, mul},
                       marked);
}

// Sinusoidal timestep embedding: timesteps * freqs
std::shared_ptr<ov::Model> make_timestep_embedding(bool marked) {
    auto timestep = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(timestep, i64_const({1}));
    auto mul = std::make_shared<v1::Multiply>(unsqueeze, f32_const({1, 128}, 0.5f));
    return make_tables(mul, {timestep}, {unsqueeze, mul}, marked);
}

struct AngleChainCase {
    std::string name;
    std::function<std::shared_ptr<ov::Model>(bool)> build;
};

class MarkSinCosAnglesToKeepInMixedPrecisionTest : public TransformationTestsF,
                                                   public testing::WithParamInterface<AngleChainCase> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AngleChainCase>& info) {
        return info.param.name;
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        model = GetParam().build(false);
        model_ref = GetParam().build(true);
        manager.register_pass<ov::pass::MarkSinCosAnglesToKeepInMixedPrecision>();
    }
};

TEST_P(MarkSinCosAnglesToKeepInMixedPrecisionTest, MarksAnglePath) {}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         MarkSinCosAnglesToKeepInMixedPrecisionTest,
                         testing::Values(AngleChainCase{"ShiftedGridRope", make_shifted_grid_rope},
                                         AngleChainCase{"CenteredGridRope", make_centered_grid_rope},
                                         AngleChainCase{"SingleAxisRope", make_single_axis_rope},
                                         AngleChainCase{"SequenceRope", make_sequence_rope},
                                         AngleChainCase{"TimestepEmbedding", make_timestep_embedding}),
                         MarkSinCosAnglesToKeepInMixedPrecisionTest::getTestCaseName);

}  // namespace

TEST_F(TransformationTestsF, MarkSinCosAnglesToKeepInMixedPrecision_SkipsMatMulActivations) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 64});
    auto matmul = std::make_shared<v0::MatMul>(x, f32_const({64, 64}, 0.1f));
    auto scaled = std::make_shared<v1::Multiply>(matmul, f32_const({}, 2.0f));
    auto sin = std::make_shared<v0::Sin>(scaled);
    model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{x});
    manager.register_pass<ov::pass::MarkSinCosAnglesToKeepInMixedPrecision>();
}

TEST_F(TransformationTestsF, MarkSinCosAnglesToKeepInMixedPrecision_SkipsConvolutionActivations) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4, -1});
    auto conv = std::make_shared<v1::Convolution>(x,
                                                  f32_const({4, 4, 3}, 0.1f),
                                                  ov::Strides{1},
                                                  ov::CoordinateDiff{1},
                                                  ov::CoordinateDiff{1},
                                                  ov::Strides{1});
    auto alpha = std::make_shared<v1::Multiply>(conv, f32_const({1, 4, 1}, 1.5f));
    auto sin = std::make_shared<v0::Sin>(alpha);
    model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{x});
    manager.register_pass<ov::pass::MarkSinCosAnglesToKeepInMixedPrecision>();
}
