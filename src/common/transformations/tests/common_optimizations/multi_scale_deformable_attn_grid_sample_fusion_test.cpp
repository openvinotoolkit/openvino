// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/grid_sample.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/msda.hpp>
#include <transformations/common_optimizations/multi_scale_deformable_attn_grid_sample_fusion.hpp>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov;

namespace {

namespace T = ov::op::v0;

// Shape constants: element type and length follow the vector, so heterogeneous
// literal lists deduce cleanly.
std::shared_ptr<T::Constant> i64_const(std::vector<int64_t> v) {
    return T::Constant::create(element::i64, ov::Shape{v.size()}, v);
}

struct PatternParams {
    size_t levels = 4;
    size_t points = 4;
    size_t heads = 2;
    size_t embed = 32;
    size_t batch = 1;
    size_t queries = 5;
    bool dynamic_value = false;
    bool align_corners = false;
    ov::op::v9::GridSample::InterpolationMode mode = ov::op::v9::GridSample::InterpolationMode::BILINEAR;
    bool foreign_second_level_value = false;
    bool wrong_normalization = false;
};

// Spatial size of level l: h = 4 + 2 * l, w = 5 + l.
size_t level_h(size_t l) {
    return 4 + 2 * l;
}
size_t level_w(size_t l) {
    return 5 + l;
}

// Builds the GridSample based multi-scale deformable attention pattern as it
// reaches the GPU plugin pipeline: VariadicSplit(value) per-level image
// chains, Gather per-level location slices, GridSample, Concat, broadcast
// Multiply and last-axis ReduceSum followed by the output projection
// Transpose([0,2,1]). Parameters: [0] value, [1] locations, [2] weights,
// [3] optional second value source.
std::shared_ptr<ov::Model> build_pattern(const PatternParams& p) {
    namespace v0 = ov::op::v0;
    namespace v1 = ov::op::v1;
    namespace v8 = ov::op::v8;
    namespace v9 = ov::op::v9;

    size_t keys = 0;
    for (size_t l = 0; l < p.levels; ++l)
        keys += level_h(l) * level_w(l);

    PartialShape value_ps(Shape{p.batch, keys, p.heads, p.embed});
    if (p.dynamic_value)
        value_ps[0] = Dimension::dynamic();
    auto value = std::make_shared<v0::Parameter>(element::f32, value_ps);
    value->set_friendly_name("value");
    auto weights =
        std::make_shared<v0::Parameter>(element::f32, Shape{p.batch, p.queries, p.heads, p.levels, p.points});
    weights->set_friendly_name("weights");

    // Locations: Add(Multiply(x, 2), -1) normalizes [0,1] locations to the
    // [-1,1] coordinates GridSample consumes.
    std::shared_ptr<v0::Parameter> loc_param =
        std::make_shared<v0::Parameter>(element::f32, Shape{p.batch, p.queries, p.heads, p.levels, p.points, 2});
    auto scale = v0::Constant::create(element::f32, Shape{1}, {p.wrong_normalization ? 3.f : 2.f});
    auto minus_one = v0::Constant::create(element::f32, Shape{1}, {-1.f});
    std::shared_ptr<ov::Node> normalized =
        std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(loc_param, scale), minus_one);
    loc_param->set_friendly_name("locations");

    ParameterVector params{value, loc_param, weights};
    std::shared_ptr<v0::Parameter> foreign_value;
    if (p.foreign_second_level_value) {
        foreign_value = std::make_shared<v0::Parameter>(element::f32, Shape{p.batch, keys, p.heads, p.embed});
        foreign_value->set_friendly_name("foreign_value");
        params.push_back(foreign_value);
    }

    std::vector<std::shared_ptr<v1::VariadicSplit>> splits;
    std::vector<int64_t> split_sizes;
    for (size_t l = 0; l < p.levels; ++l)
        split_sizes.push_back(static_cast<int64_t>(level_h(l) * level_w(l)));
    auto axis_one = T::Constant::create(element::i64, Shape{}, {1});
    splits.push_back(
        std::make_shared<v1::VariadicSplit>(value,
                                            axis_one,
                                            v0::Constant::create(element::i64, Shape{p.levels}, split_sizes)));
    if (foreign_value)
        splits.push_back(
            std::make_shared<v1::VariadicSplit>(foreign_value,
                                                axis_one,
                                                v0::Constant::create(element::i64, Shape{p.levels}, split_sizes)));

    OutputVector level_outputs;
    for (size_t l = 0; l < p.levels; ++l) {
        const size_t h = level_h(l), w = level_w(l), s = h * w;
        const auto& split = (p.foreign_second_level_value && l == 1) ? splits[1] : splits[0];

        // Image chain: Reshape -> Transpose -> Reshape -> GridSample input 0.
        auto image_flat =
            std::make_shared<v1::Reshape>(split->output(l),
                                          i64_const({int64_t(p.batch), int64_t(s), int64_t(p.heads * p.embed)}),
                                          true);
        auto image_transpose = std::make_shared<v1::Transpose>(image_flat, i64_const({0, 2, 1}));
        auto image = std::make_shared<v1::Reshape>(
            image_transpose,
            i64_const({int64_t(p.batch * p.heads), int64_t(p.embed), int64_t(h), int64_t(w)}),
            true);

        // Locations chain: Gather -> Transpose -> Reshape -> GridSample input 1.
        auto gathered = std::make_shared<v8::Gather>(normalized,
                                                     T::Constant::create(element::i64, Shape{}, {int64_t(l)}),
                                                     T::Constant::create(element::i64, Shape{}, {3}));
        auto coords_transpose = std::make_shared<v1::Transpose>(gathered, i64_const({0, 2, 1, 3, 4}));
        auto coords = std::make_shared<v1::Reshape>(
            coords_transpose,
            i64_const({int64_t(p.batch * p.heads), int64_t(p.queries), int64_t(p.points), 2}),
            true);

        v9::GridSample::Attributes attributes{p.align_corners, p.mode, v9::GridSample::PaddingMode::ZEROS};
        auto grid = std::make_shared<v9::GridSample>(image, coords, attributes);
        level_outputs.push_back(std::make_shared<v1::Reshape>(
            grid,
            i64_const({int64_t(p.batch * p.heads), int64_t(p.embed), int64_t(p.queries), 1, int64_t(p.points)}),
            true));
    }

    auto concat = std::make_shared<v0::Concat>(level_outputs, -2);
    auto values_reshape =
        std::make_shared<v1::Reshape>(concat, i64_const({0, int64_t(p.embed), 0, int64_t(p.levels * p.points)}), true);

    // Weights chain: Reshape -> Transpose -> Reshape -> broadcast operand.
    auto weights_value = std::make_shared<v1::Reshape>(
        weights,
        i64_const({int64_t(p.batch), int64_t(p.queries), int64_t(p.heads), int64_t(p.levels), int64_t(p.points)}),
        true);
    auto weights_transpose = std::make_shared<v1::Transpose>(weights_value, i64_const({0, 2, 1, 3, 4}));
    auto weights_reshape = std::make_shared<v1::Reshape>(
        weights_transpose,
        i64_const({int64_t(p.batch * p.heads), 1, int64_t(p.queries), int64_t(p.levels * p.points)}),
        true);

    auto mul = std::make_shared<v1::Multiply>(values_reshape, weights_reshape);
    auto reduce = std::make_shared<v1::ReduceSum>(mul, i64_const({-1}), false);
    auto root = std::make_shared<v1::Reshape>(reduce, i64_const({-1, int64_t(p.heads * p.embed), 0}), true);
    auto output = std::make_shared<v1::Transpose>(root, i64_const({0, 2, 1}));

    return std::make_shared<ov::Model>(OutputVector{output}, params, "MSDAGridSamplePattern");
}

std::shared_ptr<ov::op::internal::MSDA> msda_of(const std::shared_ptr<ov::Model>& model) {
    for (auto& op : model->get_ops())
        if (ov::is_type<ov::op::internal::MSDA>(op))
            return ov::as_type_ptr<ov::op::internal::MSDA>(op);
    return nullptr;
}

size_t run_fusion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::MultiScaleDeformableAttnGridSampleFusion>();
    manager.run_passes(model);
    return count_ops_of_type<ov::op::internal::MSDA>(model);
}

}  // namespace

TEST(MultiScaleDeformableAttnGridSampleFusion, four_levels_four_points) {
    PatternParams p;
    auto model = build_pattern(p);
    EXPECT_EQ(run_fusion(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::op::v9::GridSample>(model), 0);
    // The [0,1] locations parameter feeds MSDA directly.
    const auto msda = msda_of(model);
    ASSERT_NE(msda, nullptr);
    EXPECT_TRUE(ov::is_type<T::Parameter>(msda->get_input_node_ptr(3)));
}

TEST(MultiScaleDeformableAttnGridSampleFusion, three_levels_two_points) {
    PatternParams p;
    p.levels = 3;
    p.points = 2;
    p.heads = 3;
    p.embed = 16;
    auto model = build_pattern(p);
    EXPECT_EQ(run_fusion(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::op::v9::GridSample>(model), 0);
}

TEST(MultiScaleDeformableAttnGridSampleFusion, single_level) {
    PatternParams p;
    p.levels = 1;
    auto model = build_pattern(p);
    EXPECT_EQ(run_fusion(model), 1);
    EXPECT_EQ(count_ops_of_type<ov::op::v9::GridSample>(model), 0);
}

TEST(MultiScaleDeformableAttnGridSampleFusion, negative_wrong_normalization) {
    // Add(Multiply(x, 3), -1) does not map [0,1] to the GridSample coordinate
    // range, so the pattern must be rejected.
    PatternParams p;
    p.wrong_normalization = true;
    auto model = build_pattern(p);
    EXPECT_EQ(run_fusion(model), 0);
}

TEST(MultiScaleDeformableAttnGridSampleFusion, negative_dynamic_value) {
    PatternParams p;
    p.dynamic_value = true;
    auto model = build_pattern(p);
    EXPECT_EQ(run_fusion(model), 0);
    EXPECT_EQ(count_ops_of_type<ov::op::v9::GridSample>(model), p.levels);
}

TEST(MultiScaleDeformableAttnGridSampleFusion, negative_align_corners) {
    PatternParams p;
    p.align_corners = true;
    auto model = build_pattern(p);
    EXPECT_EQ(run_fusion(model), 0);
}

TEST(MultiScaleDeformableAttnGridSampleFusion, negative_nearest_mode) {
    PatternParams p;
    p.mode = ov::op::v9::GridSample::InterpolationMode::NEAREST;
    auto model = build_pattern(p);
    EXPECT_EQ(run_fusion(model), 0);
}

TEST(MultiScaleDeformableAttnGridSampleFusion, negative_mismatched_ancestry) {
    // The second level samples a different value tensor.
    PatternParams p;
    p.foreign_second_level_value = true;
    auto model = build_pattern(p);
    EXPECT_EQ(run_fusion(model), 0);
}
