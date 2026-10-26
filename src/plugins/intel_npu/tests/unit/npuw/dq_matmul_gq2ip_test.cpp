// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

using namespace ov::npuw::patterns::opt;

struct SubgraphNodes {
    std::shared_ptr<ov::op::v0::Parameter> qweight;
    std::shared_ptr<ov::op::v0::Parameter> qcoeff;
    std::shared_ptr<ov::Model> model;
};

// Build the subgraph the grouped patterns match:
//   W:[N, G, group] -> Convert(f16) -> [Subtract(0.5)] -> Multiply(S) -> Reshape -> MatMul(act, ^)
//   S:[N, G, 1] ------------------------------------------^
//   act:[1, T, G * group] ----------------------------------------------------------------^
// The patterns transpose W and S to [G, N, group] and present a sub-byte W in an i4 container.
SubgraphNodes build_grouped_subgraph(const ov::element::Type& weight_type,
                                     const ov::Shape& weight_shape,
                                     const ov::Shape& act_shape,
                                     bool zero_point = false,
                                     bool shared_zero_point = false) {
    const std::size_t n = weight_shape[0];
    const std::size_t groups = weight_shape[1];
    const std::size_t group = weight_shape[2];

    auto qweight = std::make_shared<ov::op::v0::Parameter>(weight_type, weight_shape);
    auto qcoeff = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{n, groups, 1});
    auto act = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, act_shape);

    auto cvtw = std::make_shared<ov::op::v0::Convert>(qweight, ov::element::f16);
    std::shared_ptr<ov::Node> dequant = cvtw;
    if (zero_point) {
        dequant =
            std::make_shared<ov::op::v1::Subtract>(cvtw,
                                                   ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.5f}));
    }
    auto muls = std::make_shared<ov::op::v1::Multiply>(dequant, qcoeff);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(
        muls,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     {static_cast<int64_t>(n), static_cast<int64_t>(groups * group)}),
        false);
    auto mm = std::make_shared<ov::op::v0::MatMul>(act, reshape, false, true);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(mm)};
    if (shared_zero_point) {
        // A second consumer of the chain's midpoint: the Subtract under the Convert.
        results.push_back(std::make_shared<ov::op::v0::Result>(dequant));
    }

    auto model = std::make_shared<ov::Model>(results, ov::ParameterVector{qweight, qcoeff, act});
    return {qweight, qcoeff, model};
}

// The single-token pattern: its gate is the multi-token one's `just_one(act_shape[0], act_shape[1])` negated.
void run_gq2i(std::shared_ptr<ov::Model>& model, Context& ctx) {
    ov::pass::Manager manager;
    manager.register_pass<DQMatMulGQ2i>(std::ref(ctx));
    manager.run_passes(model);
}

void run_gq2ip(std::shared_ptr<ov::Model>& model, Context& ctx) {
    ov::pass::Manager manager;
    manager.register_pass<DQMatMulGQ2iP>(std::ref(ctx));
    manager.run_passes(model);
}

TEST(DQMatMulGQ2iP, MultiTokenSubByteIsPresentedAsI4) {
    auto nodes = build_grouped_subgraph(ov::element::i2, ov::Shape{8, 2, 16}, ov::Shape{1, 4, 32});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i4);
    EXPECT_EQ(ctx.closures_to_i4.count(nodes.qweight), 1u);
    EXPECT_EQ(ctx.closures_to_permute.at(nodes.qweight), Context::Axes({1, 0, 2}));
}

TEST(DQMatMulGQ2i, SingleTokenKeepsTheStoredWidth) {
    auto nodes = build_grouped_subgraph(ov::element::i2, ov::Shape{8, 2, 16}, ov::Shape{1, 1, 32});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2i(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i2);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_EQ(ctx.closures_to_permute.at(nodes.qweight), Context::Axes({1, 0, 2}));
}

TEST(DQMatMulGQ2iP, MultiTokenGateDoesNotTakeTheSingleTokenShape) {
    auto nodes = build_grouped_subgraph(ov::element::i2, ov::Shape{8, 2, 16}, ov::Shape{1, 1, 32});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i2);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
}

// The presentation's swap is a host byte-row copy, so a group the codes do not fill byte-wise takes no
// presentation at all: the gate declines and the stored width stays.
TEST(DQMatMulGQ2iP, UnalignedGroupTakesNoPresentation) {
    auto nodes = build_grouped_subgraph(ov::element::i2, ov::Shape{8, 2, 18}, ov::Shape{1, 4, 36});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i2);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
}

TEST(DQMatMulGQ2i, UnalignedGroupTakesNoPresentation) {
    auto nodes = build_grouped_subgraph(ov::element::i2, ov::Shape{8, 2, 18}, ov::Shape{1, 4, 36});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2i(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i2);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_TRUE(ctx.closures_to_permute.empty());
}

// One closure carries one transform per parameter: registering the same order twice is a no-op, a different one
// is refused (it would compose onto the first shape and only the last order would reach the bank).
TEST(DQMatMulGQ2iP, ASecondPermutationIsRefused) {
    auto nodes = build_grouped_subgraph(ov::element::i2, ov::Shape{8, 2, 16}, ov::Shape{1, 4, 32});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    const auto shape_after_pass = nodes.qweight->get_shape();
    const auto registrations = ctx.closures_to_permute.size();

    ctx.permute(nodes.qweight, {1, 0, 2});
    EXPECT_EQ(ctx.closures_to_permute.size(), registrations);
    EXPECT_EQ(nodes.qweight->get_shape(), shape_after_pass);

    EXPECT_ANY_THROW(ctx.permute(nodes.qweight, {2, 0, 1}));
    EXPECT_EQ(nodes.qweight->get_shape(), shape_after_pass);
}

// One closure carries one scalar multiply per parameter: the same factor twice is a no-op, a different one is
// refused (it would compose onto the first and only the last would reach the bank).
TEST(DQMatMulGQ2iP, ASecondScaleIsRefused) {
    auto nodes = build_grouped_subgraph(ov::element::u1, ov::Shape{8, 2, 16}, ov::Shape{1, 4, 32}, /*zero_point=*/true);
    Context ctx;

    ctx.scale(nodes.qcoeff, 0.5f);
    ctx.scale(nodes.qcoeff, 0.5f);
    ASSERT_EQ(ctx.closures_to_scale.size(), 1u);
    EXPECT_FLOAT_EQ(ctx.closures_to_scale.at(nodes.qcoeff), 0.5f);

    EXPECT_ANY_THROW(ctx.scale(nodes.qcoeff, 0.25f));
    EXPECT_FLOAT_EQ(ctx.closures_to_scale.at(nodes.qcoeff), 0.5f);
}

TEST(DQMatMulGQ2iP, FullDequantizeRouteDoesNotWiden) {
    auto nodes = build_grouped_subgraph(ov::element::i2, ov::Shape{8, 2, 16}, ov::Shape{1, 4, 32});
    Context ctx;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i2);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_EQ(ctx.closures_to_permute.count(nodes.qweight), 1u);
}

TEST(DQMatMulGQ2iP, FourBitWeightIsNotWidened) {
    auto nodes = build_grouped_subgraph(ov::element::i4, ov::Shape{8, 2, 16}, ov::Shape{1, 4, 32});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i4);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_EQ(ctx.closures_to_permute.count(nodes.qweight), 1u);
}

// The sign grid: (code - 0.5) * 2s is code * s once the widened payload carries -1/+1, so the group scale is
// halved in the bank and the Subtract leaves the compute path.
TEST(DQMatMulGQ2iP, SignGridWideningHalvesTheScale) {
    auto nodes = build_grouped_subgraph(ov::element::u1, ov::Shape{8, 2, 16}, ov::Shape{1, 4, 32}, /*zero_point=*/true);
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i4);
    EXPECT_EQ(ctx.closures_to_i4.count(nodes.qweight), 1u);
    EXPECT_EQ(ctx.closures_to_permute.count(nodes.qweight), 1u);
    ASSERT_EQ(ctx.closures_to_scale.count(nodes.qcoeff), 1u);
    EXPECT_FLOAT_EQ(ctx.closures_to_scale.at(nodes.qcoeff), 0.5f);

    const auto& ops = nodes.model->get_ordered_ops();
    EXPECT_EQ(std::count_if(ops.begin(),
                            ops.end(),
                            [](const std::shared_ptr<ov::Node>& node) {
                                return ov::as_type_ptr<ov::op::v1::Subtract>(node) != nullptr;
                            }),
              0);
}

// Without a zero point the weight is not a sign grid: the widened -1/+1 codes would change what its 0/1 payload
// decodes to, so it keeps its stored width.
TEST(DQMatMulGQ2iP, OneBitWeightWithoutAZeroPointKeepsTheStoredWidth) {
    auto nodes = build_grouped_subgraph(ov::element::u1, ov::Shape{8, 2, 16}, ov::Shape{1, 4, 32});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::u1);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_TRUE(ctx.closures_to_scale.empty());
    EXPECT_EQ(ctx.closures_to_permute.count(nodes.qweight), 1u);
}

// The zero point another consumer still reads keeps the chain's own coding, so the weight keeps its stored
// width as well: widening it would leave that consumer with -1/+1 codes while it subtracts 0.5.
TEST(DQMatMulGQ2iP, AZeroPointAnotherConsumerReadsKeepsTheStoredWidth) {
    auto nodes = build_grouped_subgraph(ov::element::u1,
                                        ov::Shape{8, 2, 16},
                                        ov::Shape{1, 4, 32},
                                        /*zero_point=*/true,
                                        /*shared_zero_point=*/true);
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::u1);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_TRUE(ctx.closures_to_scale.empty());
}

// The unrolled presentation splits the weight per group and re-inserts the zero point under each split, so the
// same codes still decode to the same values there.
TEST(DQMatMulGQ2iP, TheUnrolledPresentationReinsertsTheZeroPoint) {
    auto nodes = build_grouped_subgraph(ov::element::u1, ov::Shape{8, 2, 16}, ov::Shape{1, 4, 32}, /*zero_point=*/true);
    Context ctx;
    ctx.mm_dq_full = true;
    run_gq2ip(nodes.model, ctx);

    int matmuls_with_zero_point = 0;
    for (const auto& node : nodes.model->get_ordered_ops()) {
        auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(node);
        if (mm && ov::as_type_ptr<ov::op::v1::Subtract>(mm->input_value(1).get_node_shared_ptr())) {
            ++matmuls_with_zero_point;
        }
    }
    // One MatMul per group split; the fixture has two groups.
    EXPECT_EQ(matmuls_with_zero_point, 2);
}

}  // namespace
