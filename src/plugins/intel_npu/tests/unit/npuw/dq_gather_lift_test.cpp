// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

using namespace ov::npuw::patterns::opt;

// The dictionary chain DQLiftGatherSymCW matches:
//   Constant(W:[Hs, N]) -> Convert(f16) -> [Subtract(0.5)] -+
//                                                            +-> Multiply -> Gather -> Result
//   Constant(S) -------------------------------------------->           ^
//   Parameter(ids) -> Convert(i32) --------------------------------------+
// The pattern lifts the Gather out of the partitioned subgraph and keeps the weight side as it is.
std::shared_ptr<ov::Model> build_gather_chain(bool zero_point) {
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{4});
    auto weight = ov::op::v0::Constant::create(ov::element::u1, ov::Shape{8, 16}, {1});
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 16}, {1.0f});
    auto indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto cvt_ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    std::shared_ptr<ov::Node> dequant = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f16);
    if (zero_point) {
        dequant =
            std::make_shared<ov::op::v1::Subtract>(dequant,
                                                   ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.5f}));
    }
    auto mul = std::make_shared<ov::op::v1::Multiply>(dequant, scale);
    auto gather = std::make_shared<ov::op::v8::Gather>(mul, cvt_ids, indices);
    auto result = std::make_shared<ov::op::v0::Result>(gather);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ids});
}

void run_lift(std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<DQLiftGatherSymCW>();
    manager.run_passes(model);
}

void run_lift_gq(std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<DQLiftGatherSymGQ>();
    manager.run_passes(model);
}

// The grouped variant of the same lift: the Reshape groups the dequantized table before the Gather, and the lift
// rebuilds that chain behind the Gathers it creates.
//   Constant(W) -> Convert(f16) -> [Subtract(0.5)] -> Multiply -> Reshape -> Convert -> Gather -> Result
//   Constant(S) -------------------------------------^                                 ^
//   Parameter(ids) -> Convert(i32) -----------------------------------------------------+
std::shared_ptr<ov::Model> build_gather_chain_gq(bool zero_point) {
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 1});
    auto weight = ov::op::v0::Constant::create(ov::element::u1, ov::Shape{2, 4, 8}, {1});
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{2, 4, 8}, {1.0f});
    auto indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto cvt_ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    std::shared_ptr<ov::Node> dequant = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f16);
    if (zero_point) {
        dequant =
            std::make_shared<ov::op::v1::Subtract>(dequant,
                                                   ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.5f}));
    }
    auto mul = std::make_shared<ov::op::v1::Multiply>(dequant, scale);
    auto reshape =
        std::make_shared<ov::op::v1::Reshape>(mul,
                                              ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 32}),
                                              false);
    auto convert = std::make_shared<ov::op::v0::Convert>(reshape, ov::element::f16);
    auto gather = std::make_shared<ov::op::v8::Gather>(convert, cvt_ids, indices);
    auto result = std::make_shared<ov::op::v0::Result>(gather);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{ids});
}

// The weight-side Gather's data input, following an optional Subtract down to its Convert.
std::shared_ptr<ov::op::v0::Convert> weight_convert_of(const std::shared_ptr<ov::op::v8::Gather>& gather,
                                                       bool& has_subtract) {
    auto data = gather->input_value(0).get_node_shared_ptr();
    auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(data);
    has_subtract = subtract != nullptr;
    auto convert =
        ov::as_type_ptr<ov::op::v0::Convert>(subtract ? subtract->input_value(0).get_node_shared_ptr() : data);
    if (convert && convert->input_value(0).get_element_type() == ov::element::u1) {
        return convert;
    }
    return nullptr;
}

// The zero point stays on the weight side of the Gather: the compiler's import fold turns `Convert(u1) ->
// Subtract(0.5)` into the weight's own two-entry table, while behind the Gather it cannot see it and a bare
// Convert has no device kernel.
TEST(DQLiftGatherSymCW, KeepsTheZeroPointOnTheWeightSide) {
    auto model = build_gather_chain(/*zero_point=*/true);
    run_lift(model);

    int gathers = 0;
    int weight_gathers = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto gather = ov::as_type_ptr<ov::op::v8::Gather>(node);
        if (!gather) {
            continue;
        }
        ++gathers;
        bool has_subtract = false;
        if (weight_convert_of(gather, has_subtract)) {
            ++weight_gathers;
            EXPECT_TRUE(has_subtract) << "the lifted weight keeps its zero point";
        }
    }
    // The lift splits the chain into a Gather on the weight and one on the scale; a pattern that does not match
    // leaves the single original Gather in place.
    EXPECT_EQ(gathers, 2);
    EXPECT_EQ(weight_gathers, 1);
}

// Without a zero point the optional operand is absent and the weight's Convert feeds the lifted Gather directly.
TEST(DQLiftGatherSymCW, WithoutAZeroPointTheConvertFeedsTheGather) {
    auto model = build_gather_chain(/*zero_point=*/false);
    run_lift(model);

    int gathers = 0;
    int weight_gathers = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto gather = ov::as_type_ptr<ov::op::v8::Gather>(node);
        if (!gather) {
            continue;
        }
        ++gathers;
        bool has_subtract = false;
        if (weight_convert_of(gather, has_subtract)) {
            ++weight_gathers;
            EXPECT_FALSE(has_subtract) << "no zero point travels with this weight";
        }
    }
    EXPECT_EQ(gathers, 2);
    EXPECT_EQ(weight_gathers, 1);
}

// The grouped lift keeps the zero point on the same side as its CW sibling.
TEST(DQLiftGatherSymGQ, KeepsTheZeroPointOnTheWeightSide) {
    auto model = build_gather_chain_gq(/*zero_point=*/true);
    run_lift_gq(model);

    int gathers = 0;
    int weight_gathers = 0;
    for (const auto& node : model->get_ordered_ops()) {
        auto gather = ov::as_type_ptr<ov::op::v8::Gather>(node);
        if (!gather) {
            continue;
        }
        ++gathers;
        bool has_subtract = false;
        if (weight_convert_of(gather, has_subtract)) {
            ++weight_gathers;
            EXPECT_TRUE(has_subtract) << "the lifted weight keeps its zero point";
        }
    }
    EXPECT_EQ(gathers, 2);
    EXPECT_EQ(weight_gathers, 1);
}

struct HostGatherChain {
    std::shared_ptr<ov::op::v0::Parameter> weight;
    std::shared_ptr<ov::op::v0::Parameter> scale;
    std::shared_ptr<ov::op::v0::Parameter> ids;
    std::shared_ptr<ov::Model> model;
};

// The dictionary chain HostGatherDQ takes: the weight and its scale are Parameters, so the dequantized table can
// only be built on the host.
//   W:[N, K] -> Convert(f16) -> [Subtract(0.5)] -> Gather -+
//                                                           +-> Multiply -> Result
//   S:[N, 1] -------------------------------> Gather -------^
//   ids:[1, T] -> Convert(i32) ----------------------^ (the Gathers' indices)
HostGatherChain build_host_gather_chain(const ov::element::Type& weight_type,
                                        bool zero_point,
                                        float zero_point_value = 0.5f) {
    const std::size_t vocab = 4;
    const std::size_t hidden = 2048;

    auto weight = std::make_shared<ov::op::v0::Parameter>(weight_type, ov::Shape{vocab, hidden});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{vocab, 1});
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1, 2});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});

    auto cvt_ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    std::shared_ptr<ov::Node> dequant = std::make_shared<ov::op::v0::Convert>(weight, ov::element::f16);
    if (zero_point) {
        dequant = std::make_shared<ov::op::v1::Subtract>(
            dequant,
            ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {zero_point_value}));
    }
    auto gather_w = std::make_shared<ov::op::v8::Gather>(dequant, cvt_ids, axis);
    auto gather_s = std::make_shared<ov::op::v8::Gather>(scale, cvt_ids, axis);
    auto mul = std::make_shared<ov::op::v1::Multiply>(gather_w, gather_s);
    auto result = std::make_shared<ov::op::v0::Result>(mul);

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{weight, scale, ids});
    return {weight, scale, ids, model};
}

void run_host_gather(std::shared_ptr<ov::Model>& model, Context& ctx) {
    ov::pass::Manager manager;
    // The pattern's result is a new Parameter that the partitioner collects into the subgraph's parameter list,
    // so the model cannot be validated between passes: it would report that parameter as undeclared, which is
    // exactly what the partitioner is about to declare.
    manager.set_per_pass_validation(false);
    manager.register_pass<HostGatherDQ>(std::ref(ctx));
    manager.run_passes(model);
}

// The host gather materialises the whole table in f16. A two-bit weight is not one the device cannot decode:
// the bank presents its codes in `i4`, so the gate leaves it off this route and the stored width stands.
TEST(HostGatherDQ, AnIntegerTwoBitDictionaryIsNotUnpackedOnTheHost) {
    auto chain = build_host_gather_chain(ov::element::i2, /*zero_point=*/false);
    Context ctx;
    run_host_gather(chain.model, ctx);

    EXPECT_TRUE(ctx.params_to_unpack.empty());
    EXPECT_FALSE(ctx.params_to_gather);
}

// The one-bit dictionary's dequantization cannot run on the device, so the whole table is unpacked on the host
// and gathered there. The gate admits the weight with its sign-grid zero point, which the reader applies.
TEST(HostGatherDQ, AOneBitWeightIsUnpackedAndGatheredOnTheHost) {
    auto chain = build_host_gather_chain(ov::element::u1, /*zero_point=*/true);
    Context ctx;
    run_host_gather(chain.model, ctx);

    ASSERT_EQ(ctx.params_to_unpack.size(), 1u);
    const auto& unpacked = ctx.params_to_unpack.begin()->first;
    const auto& unpack = ctx.params_to_unpack.begin()->second;
    EXPECT_EQ(unpack.w, chain.weight);
    EXPECT_EQ(unpack.s, chain.scale);
    EXPECT_FALSE(unpack.z);

    ASSERT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_EQ(ctx.params_to_gather->pold, unpacked);
    EXPECT_EQ(ctx.params_to_gather->pids, chain.ids);
}

// A u1 weight without the sign grid's zero point is boolean data, and a wider weight's zero point is not the
// reader's to apply: both keep their old route instead of being decoded against a zero point they do not carry.
TEST(HostGatherDQ, AOneBitWeightWithoutItsZeroPointKeepsItsRoute) {
    auto chain = build_host_gather_chain(ov::element::u1, /*zero_point=*/false);
    Context ctx;
    run_host_gather(chain.model, ctx);

    EXPECT_TRUE(ctx.params_to_unpack.empty());
    EXPECT_FALSE(ctx.params_to_gather);
}

TEST(HostGatherDQ, AWiderWeightWithAZeroPointKeepsItsRoute) {
    auto chain = build_host_gather_chain(ov::element::i4, /*zero_point=*/true);
    Context ctx;
    run_host_gather(chain.model, ctx);

    EXPECT_TRUE(ctx.params_to_unpack.empty());
    EXPECT_FALSE(ctx.params_to_gather);
}

// The reader applies 0.5, so a chain carrying another zero point is refused rather than decoded wrong.
TEST(HostGatherDQ, AnotherZeroPointIsRefused) {
    auto chain = build_host_gather_chain(ov::element::u1, /*zero_point=*/true, /*zero_point_value=*/0.25f);
    Context ctx;
    EXPECT_ANY_THROW(run_host_gather(chain.model, ctx));
}

}  // namespace
