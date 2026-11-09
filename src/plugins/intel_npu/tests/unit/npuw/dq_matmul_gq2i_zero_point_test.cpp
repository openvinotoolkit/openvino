// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/validate.hpp"
#include "partitioning/patterns/opt.hpp"
#include "util.hpp"

namespace {

using namespace ov;

constexpr size_t kOutFeatures = 4;
constexpr size_t kGroups = 2;
constexpr size_t kGroupSize = 8;

struct GroupQuantModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<op::v0::Parameter> weight;
    std::shared_ptr<op::v0::Parameter> zero_point;
    std::shared_ptr<op::v0::Parameter> scale;
};

// Param(W)[OC,G,GS] -> Convert(f16) -> Subtract(Convert(Param(Z)[OC,G,1])) -> Multiply(Param(S)[OC,G,1])
//   -> Reshape[OC,IC] -> Convert(f32) -> MatMul(Act[1,N,IC], transpose_b=true)
GroupQuantModel make_model(const element::Type zero_point_type, const size_t num_tokens, const bool rank4_act = false) {
    auto weight = std::make_shared<op::v0::Parameter>(element::u2, Shape{kOutFeatures, kGroups, kGroupSize});
    auto zero_point = std::make_shared<op::v0::Parameter>(zero_point_type, Shape{kOutFeatures, kGroups, 1});
    auto scale = std::make_shared<op::v0::Parameter>(element::f16, Shape{kOutFeatures, kGroups, 1});
    const Shape act_shape =
        rank4_act ? Shape{1, 1, num_tokens, kGroups * kGroupSize} : Shape{1, num_tokens, kGroups * kGroupSize};
    auto act = std::make_shared<op::v0::Parameter>(element::f32, act_shape);

    auto weight_cvt = std::make_shared<op::v0::Convert>(weight, element::f16);
    auto zero_point_cvt = std::make_shared<op::v0::Convert>(zero_point, element::f16);
    auto subtract = std::make_shared<op::v1::Subtract>(weight_cvt, zero_point_cvt);
    auto multiply = std::make_shared<op::v1::Multiply>(scale, subtract);
    auto reshape_shape = op::v0::Constant::create(element::i64, Shape{2}, {kOutFeatures, kGroups * kGroupSize});
    auto reshape = std::make_shared<op::v1::Reshape>(multiply, reshape_shape, false);
    auto out_cvt = std::make_shared<op::v0::Convert>(reshape, element::f32);
    auto matmul = std::make_shared<op::v0::MatMul>(act, out_cvt, false, true);

    return {std::make_shared<ov::Model>(ResultVector{std::make_shared<op::v0::Result>(matmul)},
                                        ParameterVector{weight, zero_point, scale, act}),
            weight,
            zero_point,
            scale};
}

template <class Pattern>
ov::npuw::patterns::opt::Context run_pattern(const GroupQuantModel& m, const bool mm_dq_full) {
    ov::npuw::patterns::opt::Context ctx(m.model, 0);
    ctx.mm_dq_full = mm_dq_full;
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<Pattern>(std::ref(ctx));
    rewr.run_on_model(m.model);
    ov::pass::Validate().run_on_model(m.model);
    return ctx;
}

size_t count_transposes(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& op) {
        return ov::is_type<op::v1::Transpose>(op);
    });
}

void expect_group_major_layout(const GroupQuantModel& m) {
    EXPECT_EQ(m.weight->get_shape(), (Shape{kGroups, kOutFeatures, kGroupSize}));
    EXPECT_EQ(m.zero_point->get_shape(), (Shape{kGroups, kOutFeatures, 1}));
    EXPECT_EQ(m.scale->get_shape(), (Shape{kGroups, kOutFeatures, 1}));
    EXPECT_EQ(count_transposes(m.model), 1u);
    EXPECT_EQ(m.model->get_results().front()->get_output_shape(0).back(), kOutFeatures);
}

}  // namespace

TEST(DQMatMulGQ2iZeroPointTest, PermutesU2WeightsScaleAndZeroPointForGenerate) {
    const auto m = make_model(element::u2, 1);
    const auto ctx = run_pattern<ov::npuw::patterns::opt::DQMatMulGQ2i>(m, false);

    EXPECT_EQ(ctx.closures_to_permute.size(), 3u);
    expect_group_major_layout(m);
}

TEST(DQMatMulGQ2iZeroPointTest, PermutesU2WeightsForRank4Activation) {
    const auto m = make_model(element::u2, 1, true);
    const auto ctx = run_pattern<ov::npuw::patterns::opt::DQMatMulGQ2i>(m, false);

    EXPECT_EQ(ctx.closures_to_permute.size(), 3u);
    expect_group_major_layout(m);
}

TEST(DQMatMulGQ2iZeroPointTest, KeepsGraphForRank4ActivationInFullMode) {
    const auto m = make_model(element::u2, 1, true);
    const auto ctx = run_pattern<ov::npuw::patterns::opt::DQMatMulGQ2i>(m, true);

    EXPECT_TRUE(ctx.closures_to_permute.empty());
}

TEST(DQMatMulGQ2iZeroPointTest, PermutesU2WeightsScaleAndZeroPointForPrefill) {
    const auto m = make_model(element::u2, 5);
    const auto ctx = run_pattern<ov::npuw::patterns::opt::DQMatMulGQ2iP>(m, false);

    EXPECT_EQ(ctx.closures_to_permute.size(), 3u);
    EXPECT_EQ(m.weight->get_shape(), (Shape{kGroups, kOutFeatures, kGroupSize}));
    EXPECT_EQ(m.zero_point->get_shape(), (Shape{kGroups, kOutFeatures, 1}));
    EXPECT_EQ(count_transposes(m.model), 1u);
}

TEST(DQMatMulGQ2iZeroPointTest, KeepsGraphWhenZeroPointTypeDiffersFromWeights) {
    const auto m = make_model(element::u8, 1);
    const auto ctx = run_pattern<ov::npuw::patterns::opt::DQMatMulGQ2i>(m, false);

    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(m.weight->get_shape(), (Shape{kOutFeatures, kGroups, kGroupSize}));
    EXPECT_EQ(count_transposes(m.model), 0u);
}

TEST(DQMatMulGQ2iZeroPointTest, KeepsGraphWithZeroPointInFullMode) {
    const auto m = make_model(element::u2, 1);
    const auto ctx = run_pattern<ov::npuw::patterns::opt::DQMatMulGQ2i>(m, true);

    EXPECT_TRUE(ctx.closures_to_permute.empty());
    EXPECT_EQ(m.weight->get_shape(), (Shape{kOutFeatures, kGroups, kGroupSize}));
}

namespace {

// Permutes {1,0,2} of a u2 tensor of the given shape and returns the values as a flat vector.
std::vector<uint8_t> permute_u2(const Shape& shape, const std::vector<uint8_t>& values) {
    const auto input = op::v0::Constant::create(element::u2, shape, values);
    const ov::Tensor src(input->get_element_type(), input->get_shape(), const_cast<void*>(input->get_data_ptr()));
    const ov::Tensor dst = ov::npuw::util::permute(src, {1, 0, 2});
    return op::v0::Constant(dst).cast_vector<uint8_t>();
}

std::vector<uint8_t> expected_permute(const Shape& shape, const std::vector<uint8_t>& values) {
    std::vector<uint8_t> expected(values.size());
    for (size_t p = 0; p < shape[0]; ++p) {
        for (size_t r = 0; r < shape[1]; ++r) {
            for (size_t k = 0; k < shape[2]; ++k) {
                expected[(r * shape[0] + p) * shape[2] + k] = values[(p * shape[1] + r) * shape[2] + k];
            }
        }
    }
    return expected;
}

std::vector<uint8_t> make_values(const size_t count) {
    std::vector<uint8_t> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = static_cast<uint8_t>((i * 7 + i / 3) % 4);
    }
    return values;
}

}  // namespace

TEST(NpuwPermuteU2Test, PermutesByteAlignedRows) {
    const Shape shape{3, 5, 8};
    const auto values = make_values(ov::shape_size(shape));
    EXPECT_EQ(permute_u2(shape, values), expected_permute(shape, values));
}

TEST(NpuwPermuteU2Test, PermutesSingleElementRows) {
    const Shape shape{7, 5, 1};
    const auto values = make_values(ov::shape_size(shape));
    EXPECT_EQ(permute_u2(shape, values), expected_permute(shape, values));
}
