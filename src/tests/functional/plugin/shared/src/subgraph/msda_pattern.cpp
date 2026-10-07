// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/msda_pattern.hpp"

#include <memory>
#include <sstream>
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
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"

namespace ov {
namespace test {

namespace {

std::shared_ptr<ov::op::v0::Constant> i64_const(const std::vector<int64_t>& v) {
    return ov::op::v0::Constant::create(element::i64, ov::Shape{v.size()}, v);
}

}  // namespace

std::string msda_shapes_to_string(const MSDAShapes& shapes) {
    std::ostringstream result;
    result << "levels=";
    for (size_t l = 0; l < shapes.levels.size(); ++l)
        result << (l ? "_" : "") << shapes.levels[static_cast<size_t>(l)].first << "x"
               << shapes.levels[static_cast<size_t>(l)].second;
    result << ",B=" << shapes.batch << ",Q=" << shapes.queries << ",H=" << shapes.heads << ",D=" << shapes.embed
           << ",P=" << shapes.points;
    return result.str();
}

void msda_generate_inputs(const std::shared_ptr<ov::Model>& model,
                          const std::vector<ov::Shape>& shapes,
                          std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs) {
    inputs.clear();
    const auto& params = model->inputs();
    for (size_t i = 0; i < params.size(); ++i) {
        ov::Tensor tensor(params[i].get_element_type(), shapes[i]);
        auto* data = tensor.data<float>();
        // Input 1 holds the sampling locations.
        const float lo = (i == 1) ? 0.05f : -1.f;
        const float hi = (i == 1) ? 0.95f : 1.f;
        for (size_t j = 0; j < tensor.get_size(); ++j)
            data[j] = lo + (hi - lo) * (static_cast<float>((j * 17 + i * 31) % 101) / 100.f);
        inputs[params[i].get_node_shared_ptr()] = tensor;
    }
}

std::string MSDAPattern::getTestCaseName(const testing::TestParamInfo<MSDAPatternParams>& obj) {
    const auto& [shapes, device] = obj.param;
    return msda_shapes_to_string(shapes) + ",targetDevice=" + device;
}

void MSDAPattern::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    msda_generate_inputs(function, targetInputStaticShapes, inputs);
}

void MSDAPattern::SetUp() {
    const auto& [shapes, device] = GetParam();
    targetDevice = device;
    const auto Q = static_cast<int64_t>(shapes.queries);
    const auto H = static_cast<int64_t>(shapes.heads), D = static_cast<int64_t>(shapes.embed);
    const auto P = static_cast<int64_t>(shapes.points), L = static_cast<int64_t>(shapes.levels.size());
    int64_t keys = 0;
    for (const auto& [h, w] : shapes.levels)
        keys += static_cast<int64_t>(h * w);

    const ov::Shape value_shape{shapes.batch, static_cast<size_t>(keys), shapes.heads, shapes.embed};
    const ov::Shape locations_shape{shapes.batch, shapes.queries, shapes.heads, shapes.levels.size(), shapes.points, 2};
    const ov::Shape weights_shape{shapes.batch, shapes.queries, shapes.heads, shapes.levels.size(), shapes.points};
    auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    auto locations = std::make_shared<ov::op::v0::Parameter>(element::f32, locations_shape);
    auto weights = std::make_shared<ov::op::v0::Parameter>(element::f32, weights_shape);

    const ov::Shape scalar_6d{1, 1, 1, 1, 1, 1};
    auto scaled =
        std::make_shared<ov::op::v1::Multiply>(locations, ov::op::v0::Constant::create(element::f32, scalar_6d, {2}));
    auto coords =
        std::make_shared<ov::op::v1::Add>(scaled, ov::op::v0::Constant::create(element::f32, scalar_6d, {-1}));

    ov::OutputVector level_outputs;
    int64_t start = 0;
    for (int64_t l = 0; l < L; ++l) {
        const auto h = static_cast<int64_t>(shapes.levels[static_cast<size_t>(l)].first);
        const auto w = static_cast<int64_t>(shapes.levels[static_cast<size_t>(l)].second);
        auto slice = std::make_shared<ov::op::v1::StridedSlice>(value,
                                                                i64_const({0, start}),
                                                                i64_const({0, start + h * w}),
                                                                i64_const({1, 1}),
                                                                std::vector<int64_t>{1, 0},
                                                                std::vector<int64_t>{1, 0});
        auto image_flat = std::make_shared<ov::op::v1::Reshape>(slice, i64_const({0, 0, H * D}), true);
        auto image_transpose = std::make_shared<ov::op::v1::Transpose>(image_flat, i64_const({0, 2, 1}));
        auto image = std::make_shared<ov::op::v1::Reshape>(image_transpose, i64_const({-1, D, h, w}), true);

        auto gathered = std::make_shared<ov::op::v8::Gather>(coords, i64_const({l}), i64_const({3}), 0);
        auto squeezed = std::make_shared<ov::op::v0::Squeeze>(gathered, i64_const({3}));
        auto coords_transpose = std::make_shared<ov::op::v1::Transpose>(squeezed, i64_const({0, 2, 1, 3, 4}));
        auto grid_coords = std::make_shared<ov::op::v1::Reshape>(coords_transpose, i64_const({-1, Q, P, 2}), true);

        ov::op::v9::GridSample::Attributes attributes{false,
                                                      ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                                      ov::op::v9::GridSample::PaddingMode::ZEROS};
        auto grid = std::make_shared<ov::op::v9::GridSample>(image, grid_coords, attributes);
        level_outputs.push_back(std::make_shared<ov::op::v1::Reshape>(grid, i64_const({-1, D, Q, 1, P}), false));
        start += h * w;
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(level_outputs, -2);
    auto values = std::make_shared<ov::op::v1::Reshape>(concat, i64_const({0, D, 0, L * P}), true);
    auto weights_transpose = std::make_shared<ov::op::v1::Transpose>(weights, i64_const({0, 2, 1, 3, 4}));
    auto weights_reshape = std::make_shared<ov::op::v1::Reshape>(weights_transpose, i64_const({-1, 1, 0, L * P}), true);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(values, weights_reshape);
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(weighted, i64_const({-1}), false);
    auto heads_output = std::make_shared<ov::op::v1::Reshape>(reduce, i64_const({-1, H * D, 0}), true);
    auto output = std::make_shared<ov::op::v1::Transpose>(heads_output, i64_const({0, 2, 1}));

    function = std::make_shared<ov::Model>(ov::OutputVector{output},
                                           ov::ParameterVector{value, locations, weights},
                                           "MSDAPattern");
    // The fused kernel computes the pixel coordinates as x * W - 0.5 instead of
    // GridSample's ((2 * x - 1 + 1) * W - 1) / 2 and accumulates the samples in
    // another order, which differs from the reference by f32 rounding only.
    abs_threshold = 1e-5f;
    rel_threshold = 1e-5f;
    init_input_shapes(static_shapes_to_test_representation({value_shape, locations_shape, weights_shape}));
}

}  // namespace test
}  // namespace ov
