// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/msda_pattern.hpp"

#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
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
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace test {

namespace {

std::shared_ptr<ov::op::v0::Constant> i64_const(const std::vector<int64_t>& v) {
    return ov::op::v0::Constant::create(element::i64, ov::Shape{v.size()}, v);
}

}  // namespace

std::string MSDAPattern::getTestCaseName(const testing::TestParamInfo<MSDAPatternParams>& obj) {
    const auto& [shapes, form, precision, device] = obj.param;
    std::ostringstream result;
    result << "levels=";
    for (size_t l = 0; l < shapes.levels.size(); ++l)
        result << (l ? "_" : "") << shapes.levels[l].first << "x" << shapes.levels[l].second;
    result << ",B=" << shapes.batch << ",Q=" << shapes.queries << ",H=" << shapes.heads << ",D=" << shapes.embed
           << ",P=" << shapes.points << ",form=" << (form == MSDAForm::VariadicSplit ? "VariadicSplit" : "StridedSlice")
           << ",inference_precision=" << precision << ",targetDevice=" << device;
    return result.str();
}

void MSDAPattern::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    // value and weights in [-1, 1]; locations in [-0.5, 1.5], so samples land
    // inside the levels, on their borders and fully outside them. Multiples of
    // 1/1024 in these ranges are exact in f16, so the f16 instances compare the
    // kernel arithmetic, not the rounding of the inputs.
    const std::vector<utils::InputGenerateData> data = {{-1, 2, 1024, 1}, {-0.5, 2, 1024, 2}, {-1, 2, 1024, 3}};
    const auto& params = function->inputs();
    inputs.clear();
    for (size_t i = 0; i < params.size(); ++i) {
        inputs.insert(
            {params[i].get_node_shared_ptr(),
             utils::create_and_fill_tensor(params[i].get_element_type(), targetInputStaticShapes[i], data[i])});
    }
}

void MSDAPattern::validate() {
    SubgraphBaseTest::validate();
    CheckNumberOfNodesWithType(compiledModel, "msda", 1);
}

void MSDAPattern::SetUp() {
    const auto& [shapes, form, precision, device] = GetParam();
    targetDevice = device;
    configuration[ov::hint::inference_precision.name()] = precision;
    // The inputs are exact in f16 (see generate_inputs), so the differences to
    // the f32 reference come from the arithmetic: on a B60 the maximum was
    // 1.2e-7 for f32 and 4.8e-4 for f16, the rounding of the stored f16 result.
    // Computing the sampling in f16 instead would differ by about 4e-2.
    abs_threshold = precision == ov::element::f16 ? 2e-3f : 1e-5f;
    rel_threshold = abs_threshold;

    const auto Q = static_cast<int64_t>(shapes.queries), H = static_cast<int64_t>(shapes.heads);
    const auto D = static_cast<int64_t>(shapes.embed), P = static_cast<int64_t>(shapes.points);
    const auto L = static_cast<int64_t>(shapes.levels.size());
    std::vector<int64_t> level_keys;
    int64_t keys = 0;
    for (const auto& [h, w] : shapes.levels) {
        level_keys.push_back(static_cast<int64_t>(h * w));
        keys += level_keys.back();
    }

    const ov::Shape value_shape{shapes.batch, static_cast<size_t>(keys), shapes.heads, shapes.embed};
    const ov::Shape locations_shape{shapes.batch, shapes.queries, shapes.heads, shapes.levels.size(), shapes.points, 2};
    const ov::Shape weights_shape{shapes.batch, shapes.queries, shapes.heads, shapes.levels.size(), shapes.points};
    auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, value_shape);
    auto locations = std::make_shared<ov::op::v0::Parameter>(element::f32, locations_shape);
    auto weights = std::make_shared<ov::op::v0::Parameter>(element::f32, weights_shape);

    // GridSample takes the [-1, 1] coordinates 2 * x - 1 of the [0, 1] locations x.
    auto scaled =
        std::make_shared<ov::op::v1::Multiply>(locations, ov::op::v0::Constant::create(element::f32, Shape{1}, {2.f}));
    auto coords =
        std::make_shared<ov::op::v1::Add>(scaled, ov::op::v0::Constant::create(element::f32, Shape{1}, {-1.f}));
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(
        value,
        ov::op::v0::Constant::create(element::i64, Shape{}, {1}),
        ov::op::v0::Constant::create(element::i64, Shape{level_keys.size()}, level_keys));

    ov::OutputVector level_outputs;
    int64_t start = 0;
    for (int64_t l = 0; l < L; ++l) {
        const auto& [h_size, w_size] = shapes.levels[static_cast<size_t>(l)];
        const auto h = static_cast<int64_t>(h_size), w = static_cast<int64_t>(w_size);

        ov::Output<ov::Node> level_value, level_coords;
        if (form == MSDAForm::VariadicSplit) {
            level_value = split->output(static_cast<size_t>(l));
            level_coords =
                std::make_shared<ov::op::v8::Gather>(coords,
                                                     ov::op::v0::Constant::create(element::i64, Shape{}, {l}),
                                                     ov::op::v0::Constant::create(element::i64, Shape{}, {3}));
        } else {
            level_value = std::make_shared<ov::op::v1::StridedSlice>(value,
                                                                     i64_const({0, start}),
                                                                     i64_const({0, start + h * w}),
                                                                     i64_const({1, 1}),
                                                                     std::vector<int64_t>{1, 0},
                                                                     std::vector<int64_t>{1, 0});
            auto gathered = std::make_shared<ov::op::v8::Gather>(coords, i64_const({l}), i64_const({3}), 0);
            level_coords = std::make_shared<ov::op::v0::Squeeze>(gathered, i64_const({3}));
        }
        start += h * w;

        auto image_flat = std::make_shared<ov::op::v1::Reshape>(level_value, i64_const({0, 0, H * D}), true);
        auto image_transpose = std::make_shared<ov::op::v1::Transpose>(image_flat, i64_const({0, 2, 1}));
        auto image = std::make_shared<ov::op::v1::Reshape>(image_transpose, i64_const({-1, D, h, w}), true);
        auto coords_transpose = std::make_shared<ov::op::v1::Transpose>(level_coords, i64_const({0, 2, 1, 3, 4}));
        auto grid_coords = std::make_shared<ov::op::v1::Reshape>(coords_transpose, i64_const({-1, Q, P, 2}), true);

        ov::op::v9::GridSample::Attributes attributes{false,
                                                      ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                                      ov::op::v9::GridSample::PaddingMode::ZEROS};
        auto grid = std::make_shared<ov::op::v9::GridSample>(image, grid_coords, attributes);
        level_outputs.push_back(std::make_shared<ov::op::v1::Reshape>(grid, i64_const({-1, D, Q, 1, P}), false));
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
    init_input_shapes(static_shapes_to_test_representation({value_shape, locations_shape, weights_shape}));
}

}  // namespace test
}  // namespace ov
