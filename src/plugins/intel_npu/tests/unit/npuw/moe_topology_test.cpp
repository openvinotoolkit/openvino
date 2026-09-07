// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_transformations/moe_topology.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <unordered_map>

#include "moe/moe_infer_utils.hpp"
#include "moe_transformations/apply_moe_device_routed_transforms.hpp"
#include "moe_transformations/device_routed_moe_transform.hpp"
#include "moe_transformations/moe_transformation.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "partitioning/online/group.hpp"
#include "partitioning/online/snapshot.hpp"
#include "partitioning/patterns/moe.hpp"

namespace {

enum class Routing { SOFTMAX_TOPK, TOPK_SOFTMAX, SIGMOID_BIAS, SCALED, UNNORMALIZED };

struct Graph {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::op::v12::ScatterElementsUpdate> scatter;
    std::shared_ptr<ov::op::v11::TopK> topk;
};

Graph make_moe(Routing routing, bool grouped = false, size_t tokens = 1, int64_t k = 2, bool shared = false) {
    constexpr size_t experts = 4, hidden = 8, intermediate = 16;
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, hidden});
    auto logits = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{tokens, experts});
    input->set_friendly_name("hidden");
    logits->set_friendly_name("selection_input");
    ov::Output<ov::Node> selection = logits;
    std::shared_ptr<ov::Node> probabilities;
    if (routing == Routing::SOFTMAX_TOPK || routing == Routing::SCALED) {
        probabilities = std::make_shared<ov::op::v8::Softmax>(logits, 1);
        selection = probabilities;
    } else if (routing == Routing::SIGMOID_BIAS) {
        probabilities = std::make_shared<ov::op::v0::Sigmoid>(logits);
        auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{experts}, {0.9f, -0.7f, 0.3f, 0.0f});
        selection = std::make_shared<ov::op::v1::Add>(probabilities, bias);
    }
    auto topk = std::make_shared<ov::op::v11::TopK>(selection,
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {k}),
                                                    -1,
                                                    ov::op::v11::TopK::Mode::MAX,
                                                    ov::op::v11::TopK::SortType::SORT_VALUES);
    ov::Output<ov::Node> scores = topk->output(0);
    if (routing == Routing::SIGMOID_BIAS)
        scores = std::make_shared<ov::op::v6::GatherElements>(probabilities, topk->output(1), 1);
    if (routing == Routing::TOPK_SOFTMAX) {
        scores = std::make_shared<ov::op::v8::Softmax>(scores, 1);
    } else if (routing != Routing::UNNORMALIZED) {
        ov::Output<ov::Node> denominator =
            std::make_shared<ov::op::v1::ReduceSum>(scores,
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                    true);
        if (routing == Routing::SIGMOID_BIAS)
            denominator =
                std::make_shared<ov::op::v1::Add>(denominator,
                                                  ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1e-6f}));
        scores = std::make_shared<ov::op::v1::Divide>(scores, denominator);
        if (routing == Routing::SCALED) {
            auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{experts}, {0.5f, 1.5f, 0.8f, 2.0f});
            auto selected_scales =
                std::make_shared<ov::op::v8::Gather>(scales,
                                                     topk->output(1),
                                                     ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
            scores = std::make_shared<ov::op::v1::Multiply>(scores, selected_scales);
        }
    }
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{tokens, experts}, {0.0f});
    auto indices = std::make_shared<ov::op::v0::Convert>(topk->output(1), ov::element::i32);
    auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
        zero,
        indices,
        scores,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto transposed =
        std::make_shared<ov::op::v1::Transpose>(scatter,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0}));
    auto routing_weights = std::make_shared<ov::op::v1::Reshape>(
        transposed,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<size_t>{experts, tokens, 1}),
        false);
    auto tile = std::make_shared<ov::op::v0::Tile>(
        input,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<size_t>{experts, 1}));
    auto batched = std::make_shared<ov::op::v1::Reshape>(
        tile,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<size_t>{experts, tokens, hidden}),
        false);
    auto weight = [grouped, experts](size_t rows, size_t cols, int seed) -> ov::Output<ov::Node> {
        std::vector<int8_t> values(experts * rows * cols);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<int8_t>((i * 3 + i / (rows * cols) + static_cast<size_t>(seed)) % 8) - 4;
        ov::Shape shape = grouped ? ov::Shape{experts, rows, cols / 4, 4} : ov::Shape{experts, rows, cols};
        auto compressed = ov::op::v0::Constant::create(ov::element::i4, shape, values);
        auto converted = std::make_shared<ov::op::v0::Convert>(compressed, ov::element::f32);
        shape.back() = 1;
        auto scales = ov::op::v0::Constant::create(ov::element::f32, shape, {0.0625f});
        ov::Output<ov::Node> result = std::make_shared<ov::op::v1::Multiply>(converted, scales);
        if (grouped)
            result = std::make_shared<ov::op::v1::Reshape>(
                result,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<size_t>{experts, rows, cols}),
                false);
        return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
    };
    auto gate = std::make_shared<ov::op::v0::MatMul>(batched, weight(intermediate, hidden, 1), false, true);
    auto up = std::make_shared<ov::op::v0::MatMul>(batched, weight(intermediate, hidden, 2), false, true);
    auto activation = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v4::Swish>(gate), up);
    auto down = std::make_shared<ov::op::v0::MatMul>(activation, weight(hidden, intermediate, 3), false, true);
    auto expert_output = std::make_shared<ov::op::v1::Reshape>(
        down,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<size_t>{experts, tokens, hidden}),
        false);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(expert_output, routing_weights);
    ov::Output<ov::Node> output =
        std::make_shared<ov::op::v1::ReduceSum>(weighted,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                                false);
    if (shared)
        output = std::make_shared<ov::op::v1::Add>(
            output,
            std::make_shared<ov::op::v0::MatMul>(
                input,
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{hidden, hidden}, {0.125f}),
                false,
                false));
    return {std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input, logits}), scatter, topk};
}

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& original) {
    // Model::evaluate has no sub-byte Gather or GatherElements evaluator.
    // Widen storage losslessly in a REFERENCE COPY and express the 2D
    // GatherElements as batched Gather; the graph under test stays packed INT4.
    auto model = original->clone();
    for (const auto& node : model->get_ordered_ops()) {
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            if (constant->get_element_type() == ov::element::i4)
                ov::replace_node(constant,
                                 ov::op::v0::Constant::create(ov::element::i8,
                                                              constant->get_shape(),
                                                              constant->cast_vector<int8_t>()));
        } else if (auto gather = ov::as_type_ptr<ov::op::v6::GatherElements>(node)) {
            EXPECT_EQ(gather->get_axis(), 1);
            EXPECT_EQ(gather->get_input_partial_shape(0).rank(), ov::Rank(2));
            ov::replace_node(
                gather,
                std::make_shared<ov::op::v8::Gather>(gather->input_value(0),
                                                     gather->input_value(1),
                                                     ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),
                                                     1));
        }
    }
    model->validate_nodes_and_infer_types();
    ov::TensorVector inputs;
    for (const auto& parameter : model->get_parameters()) {
        ov::Tensor tensor(ov::element::f32, parameter->get_shape());
        for (size_t i = 0; i < tensor.get_size(); ++i)
            tensor.data<float>()[i] = static_cast<float>((i * 7 + 3) % 19) / 8.0f - 1.0f;
        inputs.push_back(tensor);
    }
    ov::TensorVector outputs;
    for (const auto& output : model->outputs())
        outputs.emplace_back(output.get_element_type(), output.get_shape());
    EXPECT_TRUE(model->evaluate(outputs, inputs));
    return outputs;
}

class GenericMoETest : public ::testing::TestWithParam<std::tuple<Routing, bool, int64_t>> {};

TEST_P(GenericMoETest, DeviceRoutingPreservesScoresAndSlicesOnlySelectedExperts) {
    const auto [routing, grouped, k] = GetParam();
    auto graph = make_moe(routing, grouped, 1, k, true);
    const auto topology = ov::npuw::moe::match_batched_moe(graph.scatter);
    ASSERT_TRUE(topology);
    EXPECT_EQ(topology->num_experts, 4u);
    EXPECT_EQ(topology->num_selected, static_cast<size_t>(k));
    EXPECT_EQ(topology->scores, graph.scatter->input_value(2));
    const auto before = evaluate(graph.model);
    ASSERT_TRUE(ov::npuw::ApplyMoEDeviceRoutedTransforms().run_on_model(graph.model));
    graph.model->validate_nodes_and_infer_types();
    const auto after = evaluate(graph.model);
    ASSERT_EQ(before[0].get_shape(), after[0].get_shape());
    for (size_t i = 0; i < before[0].get_size(); ++i)
        EXPECT_NEAR(before[0].data<float>()[i], after[0].data<float>()[i], 1e-5f);
    size_t sparse_matmuls = 0;
    for (const auto& node : graph.model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::MatMul>(node) && node->get_output_shape(0).size() == 3) {
            EXPECT_EQ(node->get_output_shape(0)[0], static_cast<size_t>(k));
            ++sparse_matmuls;
        }
        EXPECT_FALSE(ov::is_type<ov::op::v12::ScatterElementsUpdate>(node));
    }
    EXPECT_EQ(sparse_matmuls, 3u);
}

INSTANTIATE_TEST_SUITE_P(RoutingSemantics,
                         GenericMoETest,
                         ::testing::Combine(::testing::Values(Routing::SOFTMAX_TOPK,
                                                              Routing::TOPK_SOFTMAX,
                                                              Routing::SIGMOID_BIAS,
                                                              Routing::SCALED,
                                                              Routing::UNNORMALIZED),
                                            ::testing::Bool(),
                                            ::testing::Values(int64_t{1}, int64_t{2}, int64_t{4})));

TEST(GenericMoETopologyTest, HostMatcherTagsSigmoidBiasAndGroupedWeightsWithoutNames) {
    auto graph = make_moe(Routing::SIGMOID_BIAS, true, 7);
    auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(graph.model);
    snapshot->buildGraph();
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<ov::npuw::patterns::moe::BatchedExpert>(snapshot, "expert");
    rewrite.run_on_model(graph.model);
    ASSERT_TRUE(graph.topk->get_rt_info().count(ov::npuw::patterns::moe::RT_INFO_MOE_K));
    EXPECT_EQ(graph.topk->get_rt_info().at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), 2u);
    auto topology = ov::npuw::moe::match_batched_moe(graph.scatter);
    ASSERT_TRUE(topology);
    EXPECT_EQ(topology->weighted_output->get_rt_info().at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), 2u);
    EXPECT_EQ(snapshot->getNodeToGroupMap()->at(topology->weighted_output)->isolatedTag(), "expert");
    EXPECT_NE(snapshot->getNodeToGroupMap()->at(topology->reduction)->isolatedTag(), "expert");
    EXPECT_FALSE(ov::npuw::pass::DeviceRoutedMoETransform().run_on_model(graph.model));
}

TEST(GenericMoETopologyTest, LegacyRouterUsesSelectionKInsteadOfAnUnrelatedScoreTopK) {
    for (const bool legacy_first : {false, true}) {
        ov::ResultVector results;
        ov::ParameterVector parameters;
        std::vector<std::shared_ptr<ov::op::v11::TopK>> selections, score_topks;
        for (const int64_t score_k : {3, 4}) {
            auto graph = make_moe(Routing::UNNORMALIZED, false, 1, 2);
            // Match GPTOSS's legacy score pattern, but keep the actual expert
            // selection TopK separate (K=2 in both layers).
            auto weight = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 8}, {0.125f}),
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}));
            auto matmul =
                std::make_shared<ov::op::v0::MatMul>(graph.model->get_parameters()[0],
                                                     std::make_shared<ov::op::v0::Convert>(weight, ov::element::f32),
                                                     false,
                                                     true);
            auto biased =
                std::make_shared<ov::op::v1::Add>(matmul,
                                                  ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, {0.0f}));
            auto score_topk = std::make_shared<ov::op::v11::TopK>(
                biased,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {score_k}),
                1,
                ov::op::v11::TopK::Mode::MAX,
                ov::op::v11::TopK::SortType::SORT_VALUES);
            score_topk->set_friendly_name("layer" + std::to_string(score_k) + ".router/score_topk");
            auto softmax = std::make_shared<ov::op::v8::Softmax>(score_topk->output(0), 1);
            auto scores =
                std::make_shared<ov::op::v8::Slice>(softmax,
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
            graph.scatter->input(2).replace_source_output(scores);
            graph.model->validate_nodes_and_infer_types();
            ASSERT_TRUE(ov::npuw::moe::match_batched_moe(graph.scatter));
            results.push_back(graph.model->get_results()[0]);
            parameters.insert(parameters.end(),
                              graph.model->get_parameters().begin(),
                              graph.model->get_parameters().end());
            selections.push_back(graph.topk);
            score_topks.push_back(score_topk);
        }
        auto model = std::make_shared<ov::Model>(results, parameters);
        auto snapshot = std::make_shared<ov::npuw::online::Snapshot>(model);
        snapshot->buildGraph();
        ov::pass::GraphRewrite rewrite;
        if (legacy_first)
            rewrite.add_matcher<ov::npuw::patterns::moe::GPTOSSRouter>(snapshot, "router");
        rewrite.add_matcher<ov::npuw::patterns::moe::BatchedExpert>(snapshot, "expert");
        if (!legacy_first)
            rewrite.add_matcher<ov::npuw::patterns::moe::GPTOSSRouter>(snapshot, "router");
        EXPECT_NO_THROW(rewrite.run_on_model(model));
        for (const auto& topk : selections)
            EXPECT_EQ(topk->get_rt_info().at(ov::npuw::patterns::moe::RT_INFO_MOE_K).as<size_t>(), 2u);
        for (const auto& topk : score_topks)
            EXPECT_FALSE(topk->get_rt_info().count(ov::npuw::patterns::moe::RT_INFO_MOE_K));
    }
}

TEST(GenericMoETopologyTest, RejectsNonzeroScatterBaseWithoutMutation) {
    auto graph = make_moe(Routing::TOPK_SOFTMAX);
    graph.scatter->input(0).replace_source_output(
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4}, {1.0f}));
    const auto original = graph.model->get_ordered_ops();
    EXPECT_FALSE(ov::npuw::pass::DeviceRoutedMoETransform().run_on_model(graph.model));
    EXPECT_EQ(original, graph.model->get_ordered_ops());
}

TEST(GenericMoETopologyTest, RejectsTopKValuesUsedAsIndices) {
    auto graph = make_moe(Routing::TOPK_SOFTMAX);
    graph.scatter->input(1).replace_source_output(
        std::make_shared<ov::op::v0::Convert>(graph.topk->output(0), ov::element::i32));
    EXPECT_FALSE(ov::npuw::moe::match_batched_moe(graph.scatter));
}

TEST(GenericMoETopologyTest, RejectsReductionScatter) {
    auto graph = make_moe(Routing::TOPK_SOFTMAX);
    graph.scatter->set_reduction(ov::op::v12::ScatterElementsUpdate::Reduction::PROD);
    EXPECT_FALSE(ov::npuw::moe::match_batched_moe(graph.scatter));
}

TEST(GenericMoETopologyTest, FullDevicePipelineAcceptsBothTopKIndexTypes) {
    for (const auto index_type : {ov::element::i32, ov::element::i64}) {
        for (const bool grouped : {false, true}) {
            SCOPED_TRACE(index_type.get_type_name());
            auto graph = make_moe(Routing::TOPK_SOFTMAX, grouped);
            graph.topk->set_index_element_type(index_type);
            graph.model->validate_nodes_and_infer_types();
            // make_moe also inserts an i32 Convert before the scatter.
            const auto before = evaluate(graph.model);
            ASSERT_TRUE(ov::npuw::ApplyMoEDeviceRoutedTransforms().run_on_model(graph.model));
            const auto after = evaluate(graph.model);
            ASSERT_EQ(before[0].get_shape(), after[0].get_shape());
            for (size_t i = 0; i < before[0].get_size(); ++i)
                EXPECT_NEAR(before[0].data<float>()[i], after[0].data<float>()[i], 1e-5f);
        }
    }
}

TEST(GenericMoETopologyTest, RejectsPDPDExpertScalesWithoutMutationEvenWhenAllExpertsAreSelected) {
    for (const int64_t k : {2, 4}) {
        auto graph = make_moe(Routing::UNNORMALIZED, false, 1, k);
        auto topology = ov::npuw::moe::match_batched_moe(graph.scatter);
        ASSERT_TRUE(topology);
        for (const auto& node : topology->expert_nodes) {
            if (auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
                auto base = ov::op::v0::Constant::create(ov::element::f32, mm->get_input_shape(1), {1.0f});
                auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, {1.0f, 2.0f, 3.0f, 4.0f});
                mm->input(1).replace_source_output(std::make_shared<ov::op::v1::Multiply>(
                    base,
                    scales,
                    ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0)));
                break;
            }
        }
        graph.model->validate_nodes_and_infer_types();
        const auto original_nodes = graph.model->get_ordered_ops();
        EXPECT_FALSE(ov::npuw::moe::match_batched_moe(graph.scatter));
        EXPECT_FALSE(ov::npuw::pass::DeviceRoutedMoETransform().run_on_model(graph.model));
        EXPECT_EQ(original_nodes, graph.model->get_ordered_ops());
    }
}

TEST(GenericMoETopologyTest, RequestedDeviceRoutingDoesNotSilentlyRunDense) {
    auto graph = make_moe(Routing::TOPK_SOFTMAX);
    graph.scatter->input(0).replace_source_output(
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 4}, {1.0f}));
    EXPECT_THROW(ov::npuw::ApplyMoEDeviceRoutedTransforms().run_on_model(graph.model), ov::Exception);
}

TEST(GenericMoETopologyTest, RejectsReductionAcrossHiddenInsteadOfExperts) {
    auto graph = make_moe(Routing::TOPK_SOFTMAX);
    auto topology = ov::npuw::moe::match_batched_moe(graph.scatter);
    ASSERT_TRUE(topology);
    topology->reduction->input(1).replace_source_output(
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    graph.model->validate_nodes_and_infer_types();
    EXPECT_FALSE(ov::npuw::moe::match_batched_moe(graph.scatter));
}

TEST(GenericMoETopologyTest, RejectsExpertViewsThatMixTokensAndFeatures) {
    auto graph = make_moe(Routing::TOPK_SOFTMAX, false, 2);
    const auto topology = ov::npuw::moe::match_batched_moe(graph.scatter);
    ASSERT_TRUE(topology);
    for (const auto& node : topology->expert_nodes) {
        if (ov::is_type<ov::op::v1::Reshape>(node) && node->input_value(0).get_node_shared_ptr() == topology->tile) {
            node->input(1).replace_source_output(
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {4, 1, 16}));
        } else if (ov::is_type<ov::op::v0::MatMul>(node)) {
            node->input(1).replace_source_output(
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 16, 16}, {0.125f}));
        }
    }
    graph.model->validate_nodes_and_infer_types();
    EXPECT_EQ(graph.model->get_output_shape(0), (ov::Shape{2, 8}));
    EXPECT_FALSE(ov::npuw::moe::match_batched_moe(graph.scatter));
}

std::shared_ptr<ov::Model> make_host_expert_model(bool grouped, size_t tokens, bool mixed_precision = false) {
    auto graph = make_moe(Routing::SIGMOID_BIAS, grouped, tokens);
    auto topology = ov::npuw::moe::match_batched_moe(graph.scatter);
    if (!topology)
        throw std::runtime_error("Test expert topology did not match");
    if (mixed_precision) {
        // Exercise a real FP16 expert arm whose output is converted to f32
        // before the final mixture, rather than a no-op Convert.
        graph.model->get_parameters()[0]->set_element_type(ov::element::f16);
        for (const auto& node : topology->expert_nodes) {
            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node))
                convert->set_destination_type(ov::element::f16);
            for (auto input : node->inputs()) {
                if (auto constant =
                        ov::as_type_ptr<ov::op::v0::Constant>(input.get_source_output().get_node_shared_ptr())) {
                    if (constant->get_element_type() == ov::element::f32)
                        input.replace_source_output(ov::op::v0::Constant::create(ov::element::f16,
                                                                                 constant->get_shape(),
                                                                                 constant->cast_vector<float>()));
                }
            }
        }
        topology->weighted_output->input(0).replace_source_output(
            std::make_shared<ov::op::v0::Convert>(topology->expert_output, ov::element::f32));
        graph.model->validate_nodes_and_infer_types();
        topology = ov::npuw::moe::match_batched_moe(graph.scatter);
        if (!topology)
            throw std::runtime_error("Mixed-precision expert topology did not match");
    }
    // Reproduce the partition boundary: hidden state, broadcast router scores,
    // and original compressed constants become parameters/closures.
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, topology->broadcast_scores.get_shape());
    scores->set_friendly_name("router_scores");
    topology->weighted_output->input(1).replace_source_output(scores);
    auto hidden = graph.model->get_parameters()[0];
    ov::ParameterVector parameters{hidden, scores};
    std::unordered_map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::op::v0::Parameter>> closures;
    for (const auto& node : topology->expert_nodes) {
        for (auto input : node->inputs()) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input.get_source_output().get_node_shared_ptr());
            if (!constant || constant->get_shape().size() < 3)
                continue;
            auto& parameter = closures[constant];
            if (!parameter) {
                parameter =
                    std::make_shared<ov::op::v0::Parameter>(constant->get_element_type(), constant->get_shape());
                parameter->set_friendly_name(constant->get_friendly_name());
                parameters.push_back(parameter);
            }
            input.replace_source_output(parameter);
        }
    }
    ov::Output<ov::Node> output = tokens == 1 ? topology->reduction->output(0) : topology->weighted_output->output(0);
    return std::make_shared<ov::Model>(ov::OutputVector{output}, parameters);
}

TEST(GenericMoETopologyTest, HostGroupedClosuresAndTopOneDecode) {
    for (const bool grouped : {false, true}) {
        for (const size_t k : {size_t{1}, size_t{2}}) {
            auto model = make_host_expert_model(grouped, 1);
            auto experts = ov::npuw::function::MoEExperts::from(model, k, 16);
            ASSERT_TRUE(experts) << "grouped=" << grouped << " k=" << k;
            ASSERT_EQ(experts->_transformed_models.size(), 1u);
            const auto& transformed = experts->_transformed_models.begin()->second;
            EXPECT_EQ(transformed->get_output_shape(0), (ov::Shape{1, 8}));
            for (const auto& entry : experts->_param_mapping)
                EXPECT_EQ(entry.second.size(), k);
        }
    }
}

TEST(GenericMoETopologyTest, PrefillTokenCountEqualToHiddenSizeDoesNotReshapeHidden) {
    auto model = make_host_expert_model(true, 8);
    auto experts = ov::npuw::function::MoEExperts::from(model, 2, 3);
    ASSERT_TRUE(experts);
    const auto& transformed = experts->_transformed_models.at(3);
    EXPECT_EQ(transformed->get_output_shape(0), (ov::Shape{1, 3, 8}));
    const auto hidden = transformed->get_parameters().at(experts->_expert_input.compiled.value());
    EXPECT_EQ(hidden->get_shape(), (ov::Shape{3, 8}));
}

TEST(GenericMoETopologyTest, HostPrefillResizesOnlyTokensAcrossAnExpertOutputConvert) {
    for (const bool grouped : {false, true}) {
        auto model = make_host_expert_model(grouped, 7, true);
        const auto original_params = model->get_parameters();
        auto experts = ov::npuw::function::MoEExperts::from(model, 2, 3);
        ASSERT_TRUE(experts);
        const auto& transformed = experts->_transformed_models.at(3);
        EXPECT_EQ(transformed->get_output_shape(0), (ov::Shape{1, 3, 8}));
        EXPECT_EQ(transformed->get_parameters().at(experts->_expert_input.compiled.value())->get_shape(),
                  (ov::Shape{3, 8}));
        ASSERT_EQ(original_params.size(), transformed->get_parameters().size());
        for (size_t i = 2; i < original_params.size(); ++i) {
            auto expected = original_params[i]->get_shape();
            expected[0] = 1;
            EXPECT_EQ(transformed->get_parameters()[i]->get_shape(), expected);
        }
    }
}

TEST(GenericMoERuntimeTest, KeepsTinySignedScoresAndSkipsOnlyExactZero) {
    ov::Tensor scores(ov::element::f32, ov::Shape{4, 1, 1});
    const std::vector<float> values{0.0f, 1e-9f, -1e-10f, 0.5f};
    std::copy(values.begin(), values.end(), scores.data<float>());
    std::map<size_t, std::vector<size_t>> tokens, experts;
    auto selected = ov::npuw::moe::parse_selected_experts_from_router(ov::get_tensor_impl(scores), 4, tokens, experts);
    EXPECT_EQ(selected, (std::vector<size_t>{1, 2, 3}));
    EXPECT_EQ(tokens.at(0), selected);
    std::fill_n(scores.data<float>(), 4, 0.0f);
    EXPECT_TRUE(
        ov::npuw::moe::parse_selected_experts_from_router(ov::get_tensor_impl(scores), 4, tokens, experts).empty());
    scores.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(ov::npuw::moe::parse_selected_experts_from_router(ov::get_tensor_impl(scores), 4, tokens, experts),
                 ov::Exception);
}

TEST(GenericMoERuntimeTest, ValidatesSubByteExpertBoundariesAndIndices) {
    ov::Tensor aligned(ov::element::i4, ov::Shape{4, 2, 4});
    EXPECT_EQ(ov::npuw::moe::slice_expert_weight(aligned, 3, 4).get_shape(), (ov::Shape{1, 2, 4}));
    EXPECT_THROW(ov::npuw::moe::slice_expert_weight(aligned, 4, 4), ov::Exception);
    ov::Tensor unaligned(ov::element::i4, ov::Shape{4, 3, 1});
    EXPECT_THROW(ov::npuw::moe::slice_expert_weight(unaligned, 1, 4), ov::Exception);
}

}  // namespace