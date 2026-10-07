// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/subgraph/msda_grid_sample_pattern.hpp"

#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace test {

namespace {

std::shared_ptr<ov::op::v0::Constant> i64_const(const std::vector<int64_t>& v) {
    return ov::op::v0::Constant::create(element::i64, ov::Shape{v.size()}, v);
}

}  // namespace

std::string MSDAGridSamplePattern::getTestCaseName(const testing::TestParamInfo<MSDAPatternParams>& obj) {
    const auto& [shapes, device] = obj.param;
    return msda_shapes_to_string(shapes) + ",targetDevice=" + device;
}

void MSDAGridSamplePattern::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    msda_generate_inputs(function, targetInputStaticShapes, inputs);
}

void MSDAGridSamplePattern::SetUp() {
    const auto& [shapes, device] = GetParam();
    targetDevice = device;
    const auto B = static_cast<int64_t>(shapes.batch), Q = static_cast<int64_t>(shapes.queries);
    const auto H = static_cast<int64_t>(shapes.heads), D = static_cast<int64_t>(shapes.embed);
    const auto P = static_cast<int64_t>(shapes.points), L = static_cast<int64_t>(shapes.levels.size());
    std::vector<int64_t> split_sizes;
    for (const auto& [h, w] : shapes.levels)
        split_sizes.push_back(static_cast<int64_t>(h * w));
    int64_t keys = 0;
    for (const auto size : split_sizes)
        keys += size;

    const ov::Shape value_shape{shapes.batch, static_cast<size_t>(keys), shapes.heads, shapes.embed};
    const ov::Shape locations_shape{shapes.batch, shapes.queries, shapes.heads, shapes.levels.size(), shapes.points, 2};
    const ov::Shape weights_shape{shapes.batch, shapes.queries, shapes.heads, shapes.levels.size(), shapes.points};
    auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    auto locations = std::make_shared<ov::op::v0::Parameter>(element::f32, locations_shape);
    auto weights = std::make_shared<ov::op::v0::Parameter>(element::f32, weights_shape);

    auto scaled =
        std::make_shared<ov::op::v1::Multiply>(locations, ov::op::v0::Constant::create(element::f32, Shape{1}, {2.f}));
    auto coords =
        std::make_shared<ov::op::v1::Add>(scaled, ov::op::v0::Constant::create(element::f32, Shape{1}, {-1.f}));
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(
        value,
        ov::op::v0::Constant::create(element::i64, Shape{}, {1}),
        ov::op::v0::Constant::create(element::i64, Shape{split_sizes.size()}, split_sizes));

    ov::OutputVector level_outputs;
    for (int64_t l = 0; l < L; ++l) {
        const auto h = static_cast<int64_t>(shapes.levels[static_cast<size_t>(l)].first);
        const auto w = static_cast<int64_t>(shapes.levels[static_cast<size_t>(l)].second);
        auto image_flat = std::make_shared<ov::op::v1::Reshape>(split->output(l), i64_const({B, h * w, H * D}), true);
        auto image_transpose = std::make_shared<ov::op::v1::Transpose>(image_flat, i64_const({0, 2, 1}));
        auto image = std::make_shared<ov::op::v1::Reshape>(image_transpose, i64_const({B * H, D, h, w}), true);

        auto gathered = std::make_shared<ov::op::v8::Gather>(coords,
                                                             ov::op::v0::Constant::create(element::i64, Shape{}, {l}),
                                                             ov::op::v0::Constant::create(element::i64, Shape{}, {3}));
        auto coords_transpose = std::make_shared<ov::op::v1::Transpose>(gathered, i64_const({0, 2, 1, 3, 4}));
        auto grid_coords = std::make_shared<ov::op::v1::Reshape>(coords_transpose, i64_const({B * H, Q, P, 2}), true);

        ov::op::v9::GridSample::Attributes attributes{false,
                                                      ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                                      ov::op::v9::GridSample::PaddingMode::ZEROS};
        auto grid = std::make_shared<ov::op::v9::GridSample>(image, grid_coords, attributes);
        level_outputs.push_back(std::make_shared<ov::op::v1::Reshape>(grid, i64_const({B * H, D, Q, 1, P}), true));
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(level_outputs, -2);
    auto values = std::make_shared<ov::op::v1::Reshape>(concat, i64_const({0, D, 0, L * P}), true);
    auto weights_value = std::make_shared<ov::op::v1::Reshape>(weights, i64_const({B, Q, H, L, P}), true);
    auto weights_transpose = std::make_shared<ov::op::v1::Transpose>(weights_value, i64_const({0, 2, 1, 3, 4}));
    auto weights_reshape =
        std::make_shared<ov::op::v1::Reshape>(weights_transpose, i64_const({B * H, 1, Q, L * P}), true);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(values, weights_reshape);
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(weighted, i64_const({-1}), false);
    auto heads_output = std::make_shared<ov::op::v1::Reshape>(reduce, i64_const({-1, H * D, 0}), true);
    auto output = std::make_shared<ov::op::v1::Transpose>(heads_output, i64_const({0, 2, 1}));

    function = std::make_shared<ov::Model>(ov::OutputVector{output},
                                           ov::ParameterVector{value, locations, weights},
                                           "MSDAGridSamplePattern");
    // The pattern reaches the fusion pass unmodified when the pipeline does not
    // insert FP16 conversions around the f32 subgraph.
    configuration[ov::hint::inference_precision.name()] = ov::element::f32;
    // The fused kernel computes the pixel coordinates as x * W - 0.5 instead of
    // GridSample's ((2 * x - 1 + 1) * W - 1) / 2 and accumulates the samples in
    // another order, which differs from the reference by f32 rounding only.
    abs_threshold = 1e-5f;
    rel_threshold = 1e-5f;
    init_input_shapes(static_shapes_to_test_representation({value_shape, locations_shape, weights_shape}));
}

}  // namespace test
}  // namespace ov
