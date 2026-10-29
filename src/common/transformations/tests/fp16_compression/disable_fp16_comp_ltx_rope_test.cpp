// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/disable_fp16_comp_ltx_rope.hpp"

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
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/squeeze.hpp"
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

// compressed frequencies; the frequency helpers append the nodes the pass marks to `expected`
std::shared_ptr<v0::Convert> f16_freqs(const ov::Shape& shape, ov::NodeVector& expected) {
    auto c = v0::Constant::create(ov::element::f16, shape, std::vector<float>(ov::shape_size(shape), 1570.8f));
    auto convert = std::make_shared<v0::Convert>(c, ov::element::f32);
    expected.push_back(convert);
    return convert;
}

// frequencies computed in the graph, as exported for LTX-2: theta^linspace * pi / 2 in f64, then cast to f32
std::shared_ptr<v0::Convert> computed_freqs(size_t count, ov::NodeVector& expected) {
    std::vector<double> exponents(count);
    for (size_t i = 0; i < count; ++i) {
        exponents[i] = count > 1 ? static_cast<double>(i) / static_cast<double>(count - 1) : 0.0;
    }
    auto theta = v0::Constant::create(ov::element::f64, ov::Shape{}, {10000.0});
    auto power =
        std::make_shared<v1::Power>(theta, v0::Constant::create(ov::element::f64, ov::Shape{count}, exponents));
    auto scaled =
        std::make_shared<v1::Multiply>(power, v0::Constant::create(ov::element::f64, ov::Shape{}, {3.14159265}));
    auto half = std::make_shared<v1::Divide>(scaled, v0::Constant::create(ov::element::f64, ov::Shape{}, {2.0}));
    auto convert = std::make_shared<v0::Convert>(half, ov::element::f32);
    expected.insert(expected.end(), {power, scaled, half, convert});
    return convert;
}

// LTX-2 transformer: the linspace length is derived from the coords shape
std::shared_ptr<v0::Convert> shape_derived_freqs(const ov::Output<ov::Node>& coords, ov::NodeVector& expected) {
    auto shape = std::make_shared<v3::ShapeOf>(coords);
    auto axes = std::make_shared<v8::Gather>(shape, i64_const({2}), i64_const({0}));
    auto count = std::make_shared<v1::Reshape>(axes, i64_const({}), false);
    auto range = std::make_shared<v4::Range>(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                             count,
                                             v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                             ov::element::f64);
    auto theta = v0::Constant::create(ov::element::f64, ov::Shape{}, {10000.0});
    auto power = std::make_shared<v1::Power>(theta, range);
    auto scaled =
        std::make_shared<v1::Multiply>(power, v0::Constant::create(ov::element::f64, ov::Shape{}, {3.14159265}));
    auto convert = std::make_shared<v0::Convert>(scaled, ov::element::f32);
    expected.insert(expected.end(), {axes, count, range, power, scaled, convert});
    return convert;
}

// Sin and Cos of the same angles, with the expected nodes marked when requested
std::shared_ptr<ov::Model> make_tables(const ov::Output<ov::Node>& angles,
                                       const ov::ParameterVector& params,
                                       ov::NodeVector marked_nodes,
                                       bool marked) {
    auto sin = std::make_shared<v0::Sin>(angles);
    auto cos = std::make_shared<v0::Cos>(angles);
    marked_nodes.push_back(sin);
    marked_nodes.push_back(cos);
    if (marked) {
        for (const auto& node : marked_nodes) {
            ov::disable_conversion(node, ov::element::f16);
        }
    }
    return std::make_shared<ov::Model>(ov::OutputVector{sin, cos}, params);
}

// LTX-Video: coords / max_pos * freqs + shift -> Transpose -> Reshape
std::shared_ptr<ov::Model> make_ltx_video_rope(bool marked) {
    auto coords = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 3, 1});
    auto grid = std::make_shared<v1::Divide>(coords, f32_const({}, 64.0f));
    auto mul = std::make_shared<v1::Multiply>(grid, f32_const({4}, 1570.8f));
    auto add = std::make_shared<v1::Add>(mul, f32_const({}, -1.0f));
    auto transpose = std::make_shared<v1::Transpose>(add, i64_const({0, 1, 3, 2}));
    auto reshape = std::make_shared<v1::Reshape>(transpose, i64_const({1, -1, 12}), false);
    return make_tables(reshape, {coords}, {grid, mul, add, transpose, reshape}, marked);
}

// LTX-2 video: (coords / max_pos * 2 - 1) * freqs -> Transpose -> Reshape
std::shared_ptr<ov::Model> make_ltx2_transposed_rope(bool marked) {
    ov::NodeVector expected;
    auto coords = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 3, 1});
    auto grid = std::make_shared<v1::Divide>(coords, f32_const({}, 64.0f));
    auto scaled = std::make_shared<v1::Multiply>(grid, f32_const({}, 2.0f));
    auto centered = std::make_shared<v1::Add>(scaled, f32_const({}, -1.0f));
    auto mul = std::make_shared<v1::Multiply>(centered, computed_freqs(4, expected));
    auto transpose = std::make_shared<v1::Transpose>(mul, i64_const({0, 1, 3, 2}));
    auto reshape = std::make_shared<v1::Reshape>(transpose, i64_const({1, -1, 12}), false);
    expected.insert(expected.end(), {grid, scaled, centered, mul, transpose, reshape});
    return make_tables(reshape, {coords}, expected, marked);
}

// LTX-2 video after MoveEltwiseUpThroughDataMov: the shift is moved above the positions Unsqueeze
std::shared_ptr<ov::Model> make_ltx2_transformer_rope(bool marked) {
    ov::NodeVector expected;
    auto coords = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 3});
    auto scaled = std::make_shared<v1::Multiply>(coords, f32_const({}, 2.0f));
    auto centered = std::make_shared<v1::Add>(scaled, f32_const({}, -1.0f));
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(centered, i64_const({-1}));
    auto mul = std::make_shared<v1::Multiply>(unsqueeze, shape_derived_freqs(coords, expected));
    auto transpose = std::make_shared<v1::Transpose>(mul, i64_const({0, 1, 3, 2}));
    auto reshape = std::make_shared<v1::Reshape>(transpose, i64_const({1, -1, 9}), false);
    expected.insert(expected.end(), {scaled, centered, unsqueeze, mul, transpose, reshape});
    return make_tables(reshape, {coords}, expected, marked);
}

// LTX-2 audio after MoveEltwiseUpThroughDataMov: the shift is moved above the position axis selection
std::shared_ptr<ov::Model> make_ltx2_selected_axis_rope(bool marked) {
    ov::NodeVector expected;
    auto coords = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1});
    auto scaled = std::make_shared<v1::Multiply>(coords, f32_const({}, 2.0f));
    auto centered = std::make_shared<v1::Add>(scaled, f32_const({}, -1.0f));
    auto squeeze = std::make_shared<v0::Squeeze>(centered, i64_const({0}));
    auto select = std::make_shared<v8::Gather>(squeeze,
                                               v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                               v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(select, i64_const({-1}));
    auto mul = std::make_shared<v1::Multiply>(unsqueeze, f16_freqs({4}, expected));
    auto swap = std::make_shared<v1::Reshape>(mul, i64_const({1, -1, 4, 1}), false);
    auto flatten = std::make_shared<v1::Reshape>(swap, i64_const({1, -1, 4}), false);
    expected.insert(expected.end(), {scaled, centered, squeeze, select, unsqueeze, mul, swap, flatten});
    return make_tables(flatten, {coords}, expected, marked);
}

// LTX-2 audio and cross-attention: a single position axis, so the transpose is exported as a Reshape
std::shared_ptr<ov::Model> make_ltx2_reshaped_rope(bool marked) {
    ov::NodeVector expected;
    auto coords = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1, 1});
    auto scaled = std::make_shared<v1::Multiply>(coords, f32_const({}, 2.0f));
    auto centered = std::make_shared<v1::Add>(scaled, f32_const({}, -1.0f));
    auto mul = std::make_shared<v1::Multiply>(centered, f16_freqs({4}, expected));
    auto swap = std::make_shared<v1::Reshape>(mul, i64_const({1, -1, 4, 1}), false);
    auto flatten = std::make_shared<v1::Reshape>(swap, i64_const({1, -1, 4}), false);
    expected.insert(expected.end(), {scaled, centered, mul, swap, flatten});
    return make_tables(flatten, {coords}, expected, marked);
}

// LTX-2 text connectors: positions from the sequence length, angles fed to Sin/Cos directly
std::shared_ptr<ov::Model> make_ltx2_connector_rope(bool marked) {
    ov::NodeVector expected;
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
    auto centered = std::make_shared<v1::Add>(scaled, f32_const({}, -1.0f));
    auto mul = std::make_shared<v1::Multiply>(centered, computed_freqs(32, expected));
    expected.insert(expected.end(), {seq_len, squeezed, range, positions, grid, unsqueeze, scaled, centered, mul});
    return make_tables(mul, {hidden}, expected, marked);
}

// LTX-2 after LinOpSequenceFusion folds the centering into the frequency Multiply
std::shared_ptr<ov::Model> make_ltx2_fused_rope(bool marked) {
    ov::NodeVector expected;
    auto coords = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1, 1});
    auto mul = std::make_shared<v1::Multiply>(coords, f16_freqs({4}, expected));
    auto shift = std::make_shared<v1::Add>(mul, f32_const({4}, -1570.8f));
    auto swap = std::make_shared<v1::Reshape>(shift, i64_const({1, -1, 4, 1}), false);
    auto flatten = std::make_shared<v1::Reshape>(swap, i64_const({1, -1, 4}), false);
    expected.insert(expected.end(), {mul, shift, swap, flatten});
    return make_tables(flatten, {coords}, expected, marked);
}

struct RopeTableCase {
    std::string name;
    std::function<std::shared_ptr<ov::Model>(bool)> build;
};

class DisableFP16CompForLtxRopeTest : public TransformationTestsF, public testing::WithParamInterface<RopeTableCase> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RopeTableCase>& info) {
        return info.param.name;
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        model = GetParam().build(false);
        model_ref = GetParam().build(true);
        manager.register_pass<ov::pass::DisableFP16CompForLtxRope>();
    }
};

TEST_P(DisableFP16CompForLtxRopeTest, MarksTableAndGridPath) {}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         DisableFP16CompForLtxRopeTest,
                         testing::Values(RopeTableCase{"LtxVideo", make_ltx_video_rope},
                                         RopeTableCase{"Ltx2Transposed", make_ltx2_transposed_rope},
                                         RopeTableCase{"Ltx2Transformer", make_ltx2_transformer_rope},
                                         RopeTableCase{"Ltx2Reshaped", make_ltx2_reshaped_rope},
                                         RopeTableCase{"Ltx2SelectedAxis", make_ltx2_selected_axis_rope},
                                         RopeTableCase{"Ltx2Connector", make_ltx2_connector_rope},
                                         RopeTableCase{"Ltx2Fused", make_ltx2_fused_rope}),
                         DisableFP16CompForLtxRopeTest::getTestCaseName);

}  // namespace

TEST_F(TransformationTestsF, DisableFP16CompForLtxRope_SkipsTimestepEmbedding) {
    auto timestep = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(timestep, i64_const({1}));
    auto mul = std::make_shared<v1::Multiply>(unsqueeze, f32_const({1, 128}, 0.5f));
    auto sin = std::make_shared<v0::Sin>(mul);
    auto cos = std::make_shared<v0::Cos>(mul);
    model = std::make_shared<ov::Model>(ov::OutputVector{sin, cos}, ov::ParameterVector{timestep});
    manager.register_pass<ov::pass::DisableFP16CompForLtxRope>();
}

TEST_F(TransformationTestsF, DisableFP16CompForLtxRope_SkipsConvolutionActivations) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4, -1});
    auto conv = std::make_shared<v1::Convolution>(x,
                                                  f32_const({4, 4, 3}, 0.1f),
                                                  ov::Strides{1},
                                                  ov::CoordinateDiff{1},
                                                  ov::CoordinateDiff{1},
                                                  ov::Strides{1});
    auto bias = std::make_shared<v1::Add>(conv, f32_const({1, 4, 1}, 0.5f));
    auto alpha = std::make_shared<v1::Multiply>(bias, f32_const({1, 4, 1}, 1.5f));
    auto sin = std::make_shared<v0::Sin>(alpha);
    model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{x});
    manager.register_pass<ov::pass::DisableFP16CompForLtxRope>();
}

TEST_F(TransformationTestsF, DisableFP16CompForLtxRope_SkipsMatMulActivations) {
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 64});
    auto matmul = std::make_shared<v0::MatMul>(x, f32_const({64, 64}, 0.1f));
    auto mul = std::make_shared<v1::Multiply>(matmul, f32_const({}, 2.0f));
    auto add = std::make_shared<v1::Add>(mul, f32_const({}, -1.0f));
    auto sin = std::make_shared<v0::Sin>(add);
    model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{x});
    manager.register_pass<ov::pass::DisableFP16CompForLtxRope>();
}

TEST_F(TransformationTestsF, DisableFP16CompForLtxRope_SkipsNonConstantFrequencies) {
    auto coords = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1, 1});
    auto scale = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{4});
    auto centered = std::make_shared<v1::Add>(coords, f32_const({}, -1.0f));
    auto mul = std::make_shared<v1::Multiply>(centered, scale);
    auto sin = std::make_shared<v0::Sin>(mul);
    model = std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{coords, scale});
    manager.register_pass<ov::pass::DisableFP16CompForLtxRope>();
}
