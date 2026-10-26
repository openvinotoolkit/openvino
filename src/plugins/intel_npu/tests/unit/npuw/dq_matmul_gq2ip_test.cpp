// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

using namespace ov::npuw::patterns::opt;

struct SubgraphNodes {
    std::shared_ptr<ov::op::v0::Parameter> qweight;
    std::shared_ptr<ov::Model> model;
};

// Build the subgraph the grouped patterns match:
//   Parameter(W:[N, G, group]) -> Convert(f16) --+
//   Parameter(S:[N, G, 1]) ---------------------- Multiply -> Reshape([N, G * group]) -> MatMul(act, ^, transpose_b)
//   Parameter(act:[1, T, G * group]) ---------------------------------------------^
// The patterns transpose W and S to [G, N, group] and present a sub-byte W in an i4 container.
SubgraphNodes build_grouped_subgraph(const ov::element::Type& weight_type,
                                     const ov::Shape& weight_shape,
                                     const ov::Shape& act_shape) {
    const std::size_t n = weight_shape[0];
    const std::size_t groups = weight_shape[1];
    const std::size_t group = weight_shape[2];

    auto qweight = std::make_shared<ov::op::v0::Parameter>(weight_type, weight_shape);
    auto qcoeff = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{n, groups, 1});
    auto act = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, act_shape);

    auto cvtw = std::make_shared<ov::op::v0::Convert>(qweight, ov::element::f16);
    auto muls = std::make_shared<ov::op::v1::Multiply>(cvtw, qcoeff);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(
        muls,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     {static_cast<int64_t>(n), static_cast<int64_t>(groups * group)}),
        false);
    auto mm = std::make_shared<ov::op::v0::MatMul>(act, reshape, false, true);
    auto result = std::make_shared<ov::op::v0::Result>(mm);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{qweight, qcoeff, act});
    return {qweight, model};
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

}  // namespace
