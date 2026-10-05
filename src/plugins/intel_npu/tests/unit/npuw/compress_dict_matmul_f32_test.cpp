// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <set>

#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/patterns/opt.hpp"

// Regression test for a segfault found on Gemma-3n (dense path, F16IC off):
// CompressDictMatMulf32 matched an f32 function input (an output of another
// subgraph) on the MatMul weight port, and Partitioner::optimize() then used it
// as a closure index (size_t underflow). Only closure Parameters may be rewritten.
// HostGather has the same closure guard for vocab Parameters.
// DQParMMGQ also guards closure Parameters before parallel MatMul merging.

namespace {

using namespace ov;

struct MatMulModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<op::v0::Parameter> act;
    std::shared_ptr<op::v0::Parameter> weight;
    std::shared_ptr<op::v0::MatMul> matmul;
    std::shared_ptr<op::v0::Result> result;
};

MatMulModel make_model(element::Type type) {
    auto act = std::make_shared<op::v0::Parameter>(type, Shape{1, 1, 8});
    auto weight = std::make_shared<op::v0::Parameter>(type, Shape{4, 8});
    auto matmul = std::make_shared<op::v0::MatMul>(act, weight, false, true);
    auto result = std::make_shared<op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{act, weight});
    return {model, act, weight, matmul, result};
}

void run_pass(const MatMulModel& graph, ov::npuw::patterns::opt::Context& ctx) {
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::CompressDictMatMulf32>(std::ref(ctx));
    rewrite.run_on_model(graph.model);
}

void expect_compressed(const MatMulModel& graph, const ov::npuw::patterns::opt::Context& ctx) {
    EXPECT_EQ(ctx.closures_to_f16.size(), 1u);
    EXPECT_EQ(ctx.closures_to_f16.count(graph.weight), 1u);
    EXPECT_EQ(graph.weight->get_element_type(), element::f16);

    auto output_cvt = ov::as_type_ptr<op::v0::Convert>(graph.result->input_value(0).get_node_shared_ptr());
    ASSERT_NE(output_cvt, nullptr);
    EXPECT_EQ(output_cvt->get_destination_type(), element::f32);

    auto new_matmul = ov::as_type_ptr<op::v0::MatMul>(output_cvt->input_value(0).get_node_shared_ptr());
    ASSERT_NE(new_matmul, nullptr);
    EXPECT_NE(new_matmul, graph.matmul);
    EXPECT_EQ(new_matmul->input_value(1).get_node_shared_ptr(), graph.weight);

    auto input_cvt = ov::as_type_ptr<op::v0::Convert>(new_matmul->input_value(0).get_node_shared_ptr());
    ASSERT_NE(input_cvt, nullptr);
    EXPECT_EQ(input_cvt->get_destination_type(), element::f16);
    EXPECT_EQ(input_cvt->input_value(0).get_node_shared_ptr(), graph.act);
    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
}

struct GatherModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<op::v0::Parameter> vocab;
    std::shared_ptr<op::v0::Parameter> ids;
    std::shared_ptr<op::v8::Gather> gather;
    std::shared_ptr<op::v0::Convert> convert;
};

GatherModel make_gather_model() {
    auto vocab = std::make_shared<op::v0::Parameter>(element::f32, Shape{16, 2048});
    auto ids = std::make_shared<op::v0::Parameter>(element::i64, Shape{1, 3});
    auto axis = op::v0::Constant::create(element::i64, Shape{}, {0});
    auto gather = std::make_shared<op::v8::Gather>(vocab, ids, axis);
    auto convert = std::make_shared<op::v0::Convert>(gather, element::f16);
    auto result = std::make_shared<op::v0::Result>(convert);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{vocab, ids});
    return {model, vocab, ids, gather, convert};
}

void run_host_gather(const GatherModel& graph, ov::npuw::patterns::opt::Context& ctx) {
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::HostGather>(std::ref(ctx));
    rewrite.run_on_model(graph.model);
}

struct ParallelMatMulModel {
    std::shared_ptr<ov::Model> model;
    ov::ParameterVector weights;
    ov::ParameterVector scales;
    std::vector<std::shared_ptr<op::v0::MatMul>> matmuls;
    ov::ResultVector results;
};

ParallelMatMulModel make_parallel_matmul_model(std::size_t branches) {
    auto act = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 8});
    auto scalar = op::v0::Constant::create(element::f32, Shape{}, {1.0f});
    auto qmmi = std::make_shared<op::v1::Multiply>(act, scalar);
    ov::ParameterVector params{act};
    ov::ParameterVector weights, scales;
    std::vector<std::shared_ptr<op::v0::MatMul>> matmuls;
    ov::ResultVector results;

    for (std::size_t i = 0; i < branches; ++i) {
        auto weight = std::make_shared<op::v0::Parameter>(element::f16, Shape{1, 8, 2});
        auto scale = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 2});
        auto weight_cvt = std::make_shared<op::v0::Convert>(weight, element::f32);
        auto scaled = std::make_shared<op::v1::Multiply>(weight_cvt, scale);
        auto shape = op::v0::Constant::create(element::i64, Shape{2}, {8, 2});
        auto reshaped = std::make_shared<op::v1::Reshape>(scaled, shape, false);
        auto matmul = std::make_shared<op::v0::MatMul>(qmmi, reshaped, false, false);
        results.push_back(std::make_shared<op::v0::Result>(matmul));
        matmuls.push_back(matmul);
        weights.push_back(weight);
        scales.push_back(scale);
        params.push_back(weight);
        params.push_back(scale);
    }

    auto model = std::make_shared<ov::Model>(results, params);
    return {model, weights, scales, matmuls, results};
}

void run_parallel_matmul(const ParallelMatMulModel& graph, ov::npuw::patterns::opt::Context& ctx) {
    ctx.pmm_dims = "2";
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::opt::DQParMMGQ>(std::ref(ctx));
    rewrite.run_on_model(graph.model);
    ov::npuw::patterns::opt::mergeParallelMatMuls(graph.model, ctx);
}

void expect_original_branch(const ParallelMatMulModel& graph, std::size_t i) {
    EXPECT_EQ(graph.results[i]->input_value(0).get_node_shared_ptr(), graph.matmuls[i]);
    auto reshape = ov::as_type_ptr<op::v1::Reshape>(graph.matmuls[i]->input_value(1).get_node_shared_ptr());
    ASSERT_NE(reshape, nullptr);
    auto scaled = ov::as_type_ptr<op::v1::Multiply>(reshape->input_value(0).get_node_shared_ptr());
    ASSERT_NE(scaled, nullptr);
    EXPECT_EQ(scaled->input_value(1).get_node_shared_ptr(), graph.scales[i]);
    auto weight_cvt = ov::as_type_ptr<op::v0::Convert>(scaled->input_value(0).get_node_shared_ptr());
    ASSERT_NE(weight_cvt, nullptr);
    EXPECT_EQ(weight_cvt->input_value(0).get_node_shared_ptr(), graph.weights[i]);
}

void expect_merged_closures(const ParallelMatMulModel& graph, const ov::npuw::patterns::opt::Context& ctx) {
    EXPECT_EQ(ctx.params_to_concat.size(), 2u);
    std::set<ov::npuw::patterns::opt::Context::PPtr> expected_weights{graph.weights[0], graph.weights[1]};
    std::set<ov::npuw::patterns::opt::Context::PPtr> expected_scales{graph.scales[0], graph.scales[1]};
    bool found_weights = false;
    bool found_scales = false;
    for (const auto& entry : ctx.params_to_concat) {
        EXPECT_EQ(entry.second.second, 2u);
        std::set<ov::npuw::patterns::opt::Context::PPtr> originals(entry.second.first.begin(),
                                                                   entry.second.first.end());
        if (entry.first->get_element_type() == element::f16) {
            EXPECT_EQ(originals, expected_weights);
            EXPECT_EQ(entry.first->get_shape(), (Shape{1, 8, 4}));
            found_weights = true;
        } else if (entry.first->get_element_type() == element::f32) {
            EXPECT_EQ(originals, expected_scales);
            EXPECT_EQ(entry.first->get_shape(), (Shape{1, 1, 4}));
            found_scales = true;
        } else {
            ADD_FAILURE() << "Unexpected concatenated Parameter type";
        }
    }
    EXPECT_TRUE(found_weights);
    EXPECT_TRUE(found_scales);
}

}  // namespace

TEST(CompressDictMatMulf32Test, NonClosureParamIsNotCompressed) {
    auto graph = make_model(element::f32);
    ov::npuw::patterns::opt::Context ctx;
    ctx.non_closure_params.insert(graph.weight);

    run_pass(graph, ctx);

    EXPECT_EQ(graph.weight->get_element_type(), element::f32);
    EXPECT_TRUE(ctx.closures_to_f16.empty());
    EXPECT_EQ(graph.result->input_value(0).get_node_shared_ptr(), graph.matmul);
    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
}

TEST(CompressDictMatMulf32Test, ClosureParamIsCompressed) {
    auto graph = make_model(element::f32);
    ov::npuw::patterns::opt::Context ctx;

    run_pass(graph, ctx);

    expect_compressed(graph, ctx);
}

TEST(CompressDictMatMulf32Test, OtherNonClosureParamDoesNotBlockClosure) {
    auto graph = make_model(element::f32);
    ov::npuw::patterns::opt::Context ctx;
    ctx.non_closure_params.insert(graph.act);

    run_pass(graph, ctx);

    expect_compressed(graph, ctx);
}

TEST(CompressDictMatMulf32Test, F16WeightIsNotTouched) {
    auto graph = make_model(element::f16);
    ov::npuw::patterns::opt::Context ctx;

    run_pass(graph, ctx);

    EXPECT_EQ(graph.weight->get_element_type(), element::f16);
    EXPECT_TRUE(ctx.closures_to_f16.empty());
    EXPECT_EQ(graph.result->input_value(0).get_node_shared_ptr(), graph.matmul);
    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
}

TEST(HostGatherClosureTest, NonClosureVocabIsNotGathered) {
    auto graph = make_gather_model();
    ov::npuw::patterns::opt::Context ctx;
    ctx.non_closure_params.insert(graph.vocab);

    run_host_gather(graph, ctx);

    EXPECT_EQ(graph.vocab->get_element_type(), element::f32);
    EXPECT_TRUE(ctx.closures_to_f16.empty());
    EXPECT_FALSE(ctx.params_to_gather.has_value());
    EXPECT_EQ(graph.convert->input_value(0).get_node_shared_ptr(), graph.gather);
    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
}

TEST(HostGatherClosureTest, ClosureVocabIsGathered) {
    auto graph = make_gather_model();
    ov::npuw::patterns::opt::Context ctx;

    run_host_gather(graph, ctx);

    EXPECT_EQ(ctx.closures_to_f16.size(), 1u);
    EXPECT_EQ(ctx.closures_to_f16.count(graph.vocab), 1u);
    EXPECT_EQ(graph.vocab->get_element_type(), element::f16);
    ASSERT_TRUE(ctx.params_to_gather.has_value());
    EXPECT_EQ(ctx.params_to_gather->pold, graph.vocab);
    EXPECT_EQ(ctx.params_to_gather->pids, graph.ids);
    EXPECT_EQ(ctx.params_to_gather->pnew->get_element_type(), element::f16);
    EXPECT_EQ(ctx.params_to_gather->pnew->get_shape(), (Shape{1, 3, 2048}));

    auto new_convert = ov::as_type_ptr<op::v0::Convert>(graph.convert->input_value(0).get_node_shared_ptr());
    ASSERT_NE(new_convert, nullptr);
    EXPECT_EQ(new_convert->get_destination_type(), element::f32);
    EXPECT_EQ(new_convert->input_value(0).get_node_shared_ptr(), ctx.params_to_gather->pnew);
    EXPECT_EQ(graph.convert->get_destination_type(), element::f16);
}

TEST(DQParMMGQClosureTest, ClosuresAreMerged) {
    auto graph = make_parallel_matmul_model(2);
    ov::npuw::patterns::opt::Context ctx;

    run_parallel_matmul(graph, ctx);

    expect_merged_closures(graph, ctx);
}

TEST(DQParMMGQClosureTest, NonClosureBranchIsNotMerged) {
    auto graph = make_parallel_matmul_model(2);
    ov::npuw::patterns::opt::Context ctx;
    ctx.non_closure_params.insert(graph.weights[1]);

    run_parallel_matmul(graph, ctx);

    EXPECT_TRUE(ctx.params_to_concat.empty());
    EXPECT_EQ(graph.weights[1]->get_element_type(), element::f16);
    expect_original_branch(graph, 0);
    expect_original_branch(graph, 1);
    EXPECT_NO_THROW(graph.model->validate_nodes_and_infer_types());
}

TEST(DQParMMGQClosureTest, ClosureBranchesStillMergeWithOneNonClosure) {
    auto graph = make_parallel_matmul_model(3);
    ov::npuw::patterns::opt::Context ctx;
    ctx.non_closure_params.insert(graph.scales[2]);

    run_parallel_matmul(graph, ctx);

    expect_merged_closures(graph, ctx);
    expect_original_branch(graph, 2);
}
