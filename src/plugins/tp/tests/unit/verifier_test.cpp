// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "l1_test_utils.hpp"
#include "openvino/op/ops.hpp"
#include "tp_test_models.hpp"
#include "tp_transformations/verifier.hpp"

namespace ov::tp::tests {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;

constexpr size_t hidden = 16;
constexpr size_t heads = 4;
constexpr size_t head_dim = 8;
constexpr size_t features = heads * head_dim;
const std::vector<Segment> by_head{{features, head_dim}};

std::shared_ptr<ov::op::v0::Parameter> input() {
    return std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, hidden});
}

std::shared_ptr<ov::op::v0::Constant> weight(ov::Shape shape, int64_t axis) {
    auto w = ov::op::v0::Constant::create(ov::element::f32, shape, std::vector<float>(ov::shape_size(shape), 0.1f));
    if (axis >= 0)
        set_shard(w, ShardAnnotation(ShardStrategy::COLUMN_PARALLEL, axis, by_head, "test"));
    return w;
}

/// x[1, S, hidden] -> [1, S, features], split over the features.
std::shared_ptr<ov::Node> column(const ov::Output<ov::Node>& x) {
    auto mm = std::make_shared<ov::op::v0::MatMul>(x, weight({features, hidden}, 0), false, true);
    mm->set_friendly_name("column");
    set_role(mm, ProjectionRole(ShardStrategy::COLUMN_PARALLEL, Collective::NONE, -1, "test"));
    return mm;
}

/// [1, S, features] split -> [1, S, hidden]; `role` false leaves the Partial sum without a TPAllReduce.
std::shared_ptr<ov::Node> row(const ov::Output<ov::Node>& x, bool role = true) {
    auto mm = std::make_shared<ov::op::v0::MatMul>(x, weight({hidden, features}, 1), false, true);
    mm->set_friendly_name("row");
    if (role)
        set_role(mm, ProjectionRole(ShardStrategy::ROW_PARALLEL, Collective::ALL_REDUCE, 0, "test"));
    return mm;
}

std::shared_ptr<ov::op::v0::Constant> shape(std::vector<int64_t> values, std::vector<size_t> annotated = {}) {
    auto c = ov::op::v0::Constant::create(ov::element::i64, {values.size()}, values);
    if (!annotated.empty()) {
        ShardAnnotation a(ShardStrategy::COLUMN_PARALLEL, -1, annotated.size() && values[annotated[0]] == heads
                                                                  ? std::vector<Segment>{{heads, 1}}
                                                                  : by_head,
                          "test");
        a.elements = annotated;
        set_shard(c, a);
    }
    return c;
}

void verify(const ov::OutputVector& results, const ov::ParameterVector& parameters, const ov::SinkVector& sinks = {}) {
    auto model = std::make_shared<ov::Model>(results, sinks, parameters);
    TPShardingVerifier(2).run_on_model(model);
}

}  // namespace

TEST(TPShardingVerifierTest, ProvesAColumnThenRowParallelPair) {
    auto x = input();
    OV_ASSERT_NO_THROW(verify({row(std::make_shared<ov::op::v4::Swish>(column(x)))}, {x}));
}

TEST(TPShardingVerifierTest, RejectsAPartialSumOutsideATPAllReduce) {
    auto x = input();
    auto act = std::make_shared<ov::op::v4::Swish>(row(column(x), /*role=*/false));
    act->set_friendly_name("act");
    OV_EXPECT_THROW(verify({act}, {x}),
                    ov::Exception,
                    AllOf(HasSubstr("failed at node \"act\" (Swish)"),
                          HasSubstr("may only enter a TPAllReduce"),
                          HasSubstr("Partial from MatMul \"row\", a contraction over a split axis")));
}

TEST(TPShardingVerifierTest, DumpsEverySplitOrAnnotatedNode) {
    auto x = input();
    auto act = std::make_shared<ov::op::v4::Swish>(column(x));
    act->set_friendly_name("act");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{row(act)}, ov::ParameterVector{x});
    std::ostringstream dump;
    TPShardingVerifier(2, &dump).run_on_model(model);

    const auto text = dump.str();
    EXPECT_THAT(text, HasSubstr("MatMul \"column\" [1,?,32]: Split(axis 2 r0:[0,16) r1:[16,32)) from Constant"));
    EXPECT_THAT(text, HasSubstr("\n    tp_role: COLUMN_PARALLEL NONE group -1 by test\n"));
    EXPECT_THAT(text, HasSubstr("Swish \"act\" [1,?,32]: Split(axis 2 r0:[0,16) r1:[16,32))"));
    // The row-parallel projection's Partial sum is Replicated past its TPAllReduce.
    EXPECT_THAT(text, HasSubstr("MatMul \"row\" [1,?,16]: Replicated\n    tp_role: ROW_PARALLEL ALL_REDUCE group 0 by test\n"));
    EXPECT_THAT(text, HasSubstr("\n    tp_shard: COLUMN_PARALLEL axis 0 segments [32/8] by test\n"));
    // Parameter and Result.
    EXPECT_THAT(text, HasSubstr("# 2 other nodes are Replicated\n# Verified\n"));
}

TEST(TPShardingVerifierTest, DumpsTheStatesUpToAFailure) {
    auto x = input();
    auto act = std::make_shared<ov::op::v4::Swish>(row(column(x), /*role=*/false));
    act->set_friendly_name("act");
    auto model = std::make_shared<ov::Model>(ov::OutputVector{act}, ov::ParameterVector{x});
    std::ostringstream dump;
    EXPECT_THROW(TPShardingVerifier(2, &dump).run_on_model(model), ov::Exception);

    const auto text = dump.str();
    EXPECT_THAT(text, HasSubstr("MatMul \"row\" [1,?,16]: Partial from MatMul \"row\""));
    EXPECT_THAT(text, HasSubstr("# Failed\n[TP] Sharding verification failed at node \"act\" (Swish)"));
    EXPECT_THAT(text, ::testing::Not(HasSubstr("# Verified")));
}

TEST(TPShardingVerifierTest, RejectsASplitModelOutput) {
    auto x = input();
    OV_EXPECT_THROW(verify({column(x)}, {x}),
                    ov::Exception,
                    AllOf(HasSubstr("a model output must be Replicated"), HasSubstr("Split(axis 2 r0:[0,16) r1:[16,32))")));
}

TEST(TPShardingVerifierTest, RejectsARowParallelRoleWithoutAPartialSum) {
    auto x = input();
    auto mm = std::make_shared<ov::op::v0::MatMul>(x, weight({hidden, hidden}, -1), false, true);
    set_role(mm, ProjectionRole(ShardStrategy::ROW_PARALLEL, Collective::ALL_REDUCE, 0, "test"));
    OV_EXPECT_THROW(verify({mm}, {x}), ov::Exception, HasSubstr("must produce a Partial sum"));
}

TEST(TPShardingVerifierTest, ProvesAnnotatedHeadReshapes) {
    auto x = input();
    auto split_heads = std::make_shared<ov::op::v1::Reshape>(column(x), shape({0, 0, heads, head_dim}, {2}), true);
    auto act = std::make_shared<ov::op::v4::Swish>(split_heads);
    auto merge = std::make_shared<ov::op::v1::Reshape>(act, shape({0, 0, features}, {2}), true);
    OV_ASSERT_NO_THROW(verify({row(merge)}, {x}));
}

TEST(TPShardingVerifierTest, RejectsAGlobalHeadCountInAShape) {
    auto x = input();
    auto split_heads = std::make_shared<ov::op::v1::Reshape>(column(x), shape({0, 0, heads, head_dim}), true);
    split_heads->set_friendly_name("heads");
    auto merge = std::make_shared<ov::op::v1::Reshape>(split_heads, shape({0, 0, features}, {2}), true);
    OV_EXPECT_THROW(verify({row(merge)}, {x}),
                    ov::Exception,
                    AllOf(HasSubstr("\"heads\" (Reshape)"), HasSubstr("holds 4 for the split output axis 2")));
}

TEST(TPShardingVerifierTest, RejectsMergingTheSplitAxisBehindAnother) {
    // [1, S, heads, head_dim] -> [1, S * heads, head_dim]: the split no longer is one axis.
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, hidden});
    auto split_heads = std::make_shared<ov::op::v1::Reshape>(column(x), shape({1, 3, heads, head_dim}, {2}), false);
    auto merged = std::make_shared<ov::op::v1::Reshape>(split_heads, shape({1, -1, head_dim}), false);
    OV_EXPECT_THROW(verify({merged}, {x}), ov::Exception, HasSubstr("does not map to one output axis"));
}

TEST(TPShardingVerifierTest, RejectsAReductionOverTheSplitAxis) {
    auto x = input();
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(column(x), axes, true);
    OV_EXPECT_THROW(verify({mean}, {x}), ov::Exception, HasSubstr("a reduction over the split axis 2"));
}

TEST(TPShardingVerifierTest, ProvesANormOverTheHeadDimension) {
    auto x = input();
    auto split_heads = std::make_shared<ov::op::v1::Reshape>(column(x), shape({0, 0, heads, head_dim}, {2}), true);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mean = std::make_shared<ov::op::v1::ReduceMean>(split_heads, axes, true);
    auto norm = std::make_shared<ov::op::v1::Divide>(split_heads, mean);
    auto scale = ov::op::v0::Constant::create(ov::element::f32, {head_dim}, std::vector<float>(head_dim, 1.f));
    auto scaled = std::make_shared<ov::op::v1::Multiply>(norm, scale);
    auto merge = std::make_shared<ov::op::v1::Reshape>(scaled, shape({0, 0, features}, {2}), true);
    OV_ASSERT_NO_THROW(verify({row(merge)}, {x}));
}

TEST(TPShardingVerifierTest, ProvesTheSizeOfAnotherAxis) {
    // The causal mask: ShapeOf of a split tensor, its sequence length.
    auto x = input();
    auto projected = column(x);
    auto sizes = std::make_shared<ov::op::v3::ShapeOf>(projected);
    auto index = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto length = std::make_shared<ov::op::v8::Gather>(sizes, index, axis);
    auto zero = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto positions = std::make_shared<ov::op::v4::Range>(zero, length, one, ov::element::i64);
    OV_ASSERT_NO_THROW(verify({row(projected), positions}, {x}));
}

TEST(TPShardingVerifierTest, RejectsTheSizeOfTheSplitAxisAsData) {
    auto x = input();
    auto projected = column(x);
    auto sizes = std::make_shared<ov::op::v3::ShapeOf>(projected);
    auto index = ov::op::v0::Constant::create(ov::element::i64, {}, {2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto width = std::make_shared<ov::op::v8::Gather>(sizes, index, axis);
    auto as_float = std::make_shared<ov::op::v0::Convert>(width, ov::element::f32);
    auto scaled = std::make_shared<ov::op::v1::Multiply>(x, as_float);
    scaled->set_friendly_name("scaled");
    OV_EXPECT_THROW(verify({row(projected), scaled}, {x}),
                    ov::Exception,
                    AllOf(HasSubstr("\"scaled\" (Multiply)"), HasSubstr("rank-dependent sizes")));
}

TEST(TPShardingVerifierTest, RejectsAnOpWithoutARule) {
    auto x = input();
    auto k = ov::op::v0::Constant::create(ov::element::i64, {}, {2});
    auto top = std::make_shared<ov::op::v11::TopK>(column(x), k, 1, "max", "value");
    OV_EXPECT_THROW(verify({top->output(0)}, {x}), ov::Exception, HasSubstr("no sharding rule for this op type"));
}

TEST(TPShardingVerifierTest, RejectsAnAssignThatWritesAnotherState) {
    auto info = ov::op::util::VariableInfo{ov::PartialShape{1, heads, -1, head_dim}, ov::element::f32, "cache"};
    auto variable = std::make_shared<ov::op::util::Variable>(info);
    auto read = std::make_shared<ov::op::v6::ReadValue>(variable);
    set_shard(read, ShardAnnotation(ShardStrategy::COLUMN_PARALLEL, 1, {{heads, 1}}, "test"));
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, heads, -1, head_dim});
    auto assign = std::make_shared<ov::op::v6::Assign>(x, variable);
    // The cache length, read from the split cache.
    auto sizes = std::make_shared<ov::op::v3::ShapeOf>(read);
    auto index = ov::op::v0::Constant::create(ov::element::i64, {}, {2});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto length = std::make_shared<ov::op::v8::Gather>(sizes, index, axis);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{x, length}, ov::SinkVector{assign}, ov::ParameterVector{x});
    OV_EXPECT_THROW(TPShardingVerifier(2).run_on_model(model),
                    ov::Exception,
                    HasSubstr("what one inference writes must be what the next one reads"));
}

// The analyzer runs the verifier: an export whose computed shape holds the global feature count
// in front of the O projection fails the compilation instead of producing wrong output.
TEST(TPShardingVerifierTest, FailsTheAnalysisOfAnUnadjustedComputedShape) {
    BlockConfig config;
    auto model = make_transformer_block(config);
    bool replaced = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v0::MatMul>(node) || !ov::is_type<ov::op::v1::Reshape>(node->get_input_node_ptr(0)))
            continue;
        auto reshape = node->get_input_node_shared_ptr(0);
        auto copied = ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 0});
        auto features = ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(config.num_heads * config.head_dim)});
        reshape->input(1).replace_source_output(std::make_shared<ov::op::v0::Concat>(ov::OutputVector{copied, features}, 0));
        replaced = true;
        break;
    }
    ASSERT_TRUE(replaced);
    model->validate_nodes_and_infer_types();
    const auto q_features = std::to_string(config.num_heads * config.head_dim);
    OV_EXPECT_THROW(analyze(model, 2),
                    ov::Exception,
                    AllOf(HasSubstr("Sharding verification failed"), HasSubstr("holds " + q_features + " for the split output axis 2")));
}

// A fused gate_up_proj split with the global lengths would hand rank 0 only gate features and
// rank 1 only up features.
TEST(TPShardingVerifierTest, RejectsAFusedSplitWithGlobalLengths) {
    BlockConfig config;
    config.gate_up = Fused::VARIADIC_SPLIT;
    auto model = make_transformer_block(config);
    analyze(model, 2);
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v1::VariadicSplit>(node))
            node->get_input_node_ptr(2)->get_rt_info().erase(ShardAnnotation::get_type_info_static());
    }
    OV_EXPECT_THROW(TPShardingVerifier(2).run_on_model(model),
                    ov::Exception,
                    AllOf(HasSubstr("(VariadicSplit)"), HasSubstr("is not its share of the global part")));
}

}  // namespace ov::tp::tests
