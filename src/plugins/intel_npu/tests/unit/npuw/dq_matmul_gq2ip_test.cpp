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

// Build the subgraph DQMatMulGQ2iP matches:
//   Parameter(W:[N, G, group]) -> Convert(f16) --+
//   Parameter(S:[N, G, 1]) ---------------------- Multiply -> Reshape([N, G * group]) -> MatMul(act, ^, transpose_b)
//   Parameter(act:[1, T, G * group]) ---------------------------------------------^
// The pattern transposes W and S to [G, N, group] and presents a sub-byte W in an i4 container.
SubgraphNodes build_gq2ip_subgraph(const ov::element::Type& weight_type, const ov::Shape& act_shape) {
    auto qweight = std::make_shared<ov::op::v0::Parameter>(weight_type, ov::Shape{8, 2, 16});
    auto qcoeff = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{8, 2, 1});
    auto act = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, act_shape);

    auto cvtw = std::make_shared<ov::op::v0::Convert>(qweight, ov::element::f16);
    auto muls = std::make_shared<ov::op::v1::Multiply>(cvtw, qcoeff);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(
        muls,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {8, 32}),
        false);
    auto mm = std::make_shared<ov::op::v0::MatMul>(act, reshape, false, true);
    auto result = std::make_shared<ov::op::v0::Result>(mm);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{qweight, qcoeff, act});
    return {qweight, model};
}

void run_gq2ip(std::shared_ptr<ov::Model>& model, Context& ctx) {
    ov::pass::Manager manager;
    manager.register_pass<DQMatMulGQ2iP>(std::ref(ctx));
    manager.run_passes(model);
}

TEST(DQMatMulGQ2iP, MultiTokenSubByteIsPresentedAsI4) {
    for (const auto& weight_type : {ov::element::Type_t::i2}) {
        auto nodes = build_gq2ip_subgraph(weight_type, {1, 4, 32});
        Context ctx;
        ctx.mm_dq_full = false;
        run_gq2ip(nodes.model, ctx);

        EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i4);
        EXPECT_EQ(ctx.closures_to_i4.count(nodes.qweight), 1u);
        EXPECT_EQ(ctx.closures_to_permute.at(nodes.qweight), Context::Axes({1, 0, 2}));
    }
}

TEST(DQMatMulGQ2iP, SingleTokenKeepsTheStoredWidth) {
    for (const auto& weight_type : {ov::element::Type_t::i2}) {
        auto nodes = build_gq2ip_subgraph(weight_type, {1, 1, 32});
        Context ctx;
        ctx.mm_dq_full = false;
        run_gq2ip(nodes.model, ctx);

        EXPECT_EQ(nodes.qweight->get_element_type(), weight_type);
        EXPECT_TRUE(ctx.closures_to_i4.empty());
    }
}

TEST(DQMatMulGQ2iP, FullDequantizeRouteDoesNotWiden) {
    auto nodes = build_gq2ip_subgraph(ov::element::Type_t::i2, {1, 4, 32});
    Context ctx;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i2);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_EQ(ctx.closures_to_permute.count(nodes.qweight), 1u);
}

TEST(DQMatMulGQ2iP, FourBitWeightIsNotWidened) {
    auto nodes = build_gq2ip_subgraph(ov::element::Type_t::i4, {1, 4, 32});
    Context ctx;
    ctx.mm_dq_full = false;
    run_gq2ip(nodes.model, ctx);

    EXPECT_EQ(nodes.qweight->get_element_type(), ov::element::i4);
    EXPECT_TRUE(ctx.closures_to_i4.empty());
    EXPECT_EQ(ctx.closures_to_permute.count(nodes.qweight), 1u);
}

}  // namespace
