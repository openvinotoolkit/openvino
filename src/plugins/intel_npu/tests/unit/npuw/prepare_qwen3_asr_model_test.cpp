// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qwen3asr/prepare_qwen3_asr_model.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "llm_stored_tokens_state.hpp"
#include "llm_test_helpers.hpp"
#include "npuw_transformations/reshape_to_static.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/openvino.hpp"
#include "qwen3asr/qwen3_asr_infer_request.hpp"
#include "util.hpp"

namespace {

std::shared_ptr<ov::Model> make_decoder_mask_model(bool convert_keys = false,
                                                   size_t branches = 1,
                                                   bool rewritten = false,
                                                   bool prefill = false,
                                                   bool fixed_position_shape = false) {
    using namespace ov::op;
    auto ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto past = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 1});
    past->output(0).set_names({"past_key_values.0.key"});
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto two = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    auto past_length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(past), two, zero);
    auto query_length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(ids), one, zero);
    auto total_length = std::make_shared<v1::Add>(past_length, query_length);
    auto keys = std::make_shared<v4::Range>(zero, total_length, one, ov::element::i64);
    auto queries = std::make_shared<v4::Range>(past_length, total_length, one, ov::element::i64);
    auto keys_u1 = std::make_shared<v0::Unsqueeze>(keys, zero);
    auto keys_u2 = std::make_shared<v0::Unsqueeze>(keys_u1, zero);
    ov::Output<ov::Node> key_positions = std::make_shared<v0::Unsqueeze>(keys_u2, zero);
    ov::Output<ov::Node> query_positions =
        std::make_shared<v0::Unsqueeze>(queries, v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 3}));
    if (convert_keys) {
        key_positions = std::make_shared<v0::Convert>(key_positions, ov::element::f32);
        query_positions = std::make_shared<v0::Convert>(query_positions, ov::element::f32);
    }

    ov::ParameterVector parameters{ids, past};
    std::shared_ptr<v0::Parameter> mask;
    std::shared_ptr<v0::Parameter> position;
    if (rewritten) {
        mask = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
        mask->set_friendly_name("attention_mask");
        mask->output(0).set_names({"attention_mask"});
        position =
            std::make_shared<v0::Parameter>(ov::element::i64, prefill ? ov::PartialShape{-1} : ov::PartialShape{1});
        position->set_friendly_name("position_ids");
        position->output(0).set_names({"position_ids"});
        parameters.push_back(mask);
        parameters.push_back(position);
    }

    ov::OutputVector outputs;
    for (size_t branch = 0; branch < branches; ++branch) {
        auto causal = std::make_shared<v1::LessEqual>(key_positions, query_positions);
        const auto causal_name = "causal_" + std::to_string(branch);
        causal->set_friendly_name(rewritten ? causal_name + "/causal" : causal_name);
        ov::Output<ov::Node> condition = causal;
        if (rewritten) {
            auto c0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            auto c1 = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
            auto c2 = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
            auto u1 = std::make_shared<v0::Unsqueeze>(mask, c1);
            auto u2 = std::make_shared<v0::Unsqueeze>(u1, c2);
            auto equal = std::make_shared<v1::Equal>(u2, c0);
            auto combined = std::make_shared<v1::LogicalAnd>(causal, equal);
            combined->set_friendly_name(causal_name);
            condition = combined;
        }
        auto scores = std::make_shared<v1::Select>(condition,
                                                   v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}),
                                                   v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0e9f}));
        outputs.push_back(std::make_shared<v8::Softmax>(scores, -1));
        const int64_t last_dim = fixed_position_shape && !(rewritten && prefill) ? 1 : -1;
        const auto shape = v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 1, last_dim});
        const ov::Output<ov::Node> position_source = rewritten ? position->output(0) : queries->output(0);
        outputs.push_back(std::make_shared<v1::Reshape>(position_source, shape, false));
    }
    return std::make_shared<ov::Model>(outputs, parameters);
}

std::shared_ptr<ov::Model> make_stateful_qwen_decoder() {
    using namespace ov::op;
    auto model = ov::test::npuw::build_llm_test_model();
    const auto ids = model->input("input_ids");
    const auto beam = model->input("beam_idx");
    std::shared_ptr<ov::Node> cache;
    for (const auto& target : beam.get_target_inputs()) {
        if (ov::is_type<ov::op::util::GatherBase>(target.get_node()->shared_from_this())) {
            cache = target.get_node()->shared_from_this();
            break;
        }
    }
    OPENVINO_ASSERT(cache, "Synthetic decoder has no beam-gathered KV cache.");
    const auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    const auto two = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    const auto past = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(cache), two, zero);
    const auto query_length = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(ids), one, zero);
    const auto total = std::make_shared<v1::Add>(past, query_length);
    const auto keys = std::make_shared<v4::Range>(zero, total, one, ov::element::i64);
    const auto queries = std::make_shared<v4::Range>(past, total, one, ov::element::i64);
    const auto k1 = std::make_shared<v0::Unsqueeze>(keys, zero);
    const auto k2 = std::make_shared<v0::Unsqueeze>(k1, zero);
    const auto k3 = std::make_shared<v0::Unsqueeze>(k2, zero);
    const auto q =
        std::make_shared<v0::Unsqueeze>(queries, v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 3}));
    const auto causal = std::make_shared<v1::LessEqual>(k3, q);
    const auto scores = std::make_shared<v1::Select>(causal,
                                                     v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}),
                                                     v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0e9f}));
    bool replaced_mask = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "model.mask_4d") {
            ov::replace_node(node, scores);
            replaced_mask = true;
            break;
        }
    }
    OPENVINO_ASSERT(replaced_mask, "Synthetic decoder has no full attention mask.");
    const auto old_mask = ov::as_type_ptr<v0::Parameter>(model->input("attention_mask").get_node_shared_ptr());
    model->remove_parameter(old_mask);
    const auto old_position = ov::as_type_ptr<v0::Parameter>(model->input("position_ids").get_node_shared_ptr());
    const auto positions =
        std::make_shared<v1::Reshape>(queries, v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, -1}), false);
    ov::replace_node(old_position, positions);
    model->remove_parameter(old_position);
    auto encoder = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 64});
    encoder->output(0).set_names({"encoder_hidden_states"});
    model->add_parameters({encoder});
    model->validate_nodes_and_infer_types();
    return model;
}

void make_static(const std::shared_ptr<ov::Model>& model, size_t query_length, size_t past_length) {
    std::map<std::string, ov::PartialShape> shapes{
        {"input_ids", {1, static_cast<int64_t>(query_length)}},
        {"past_key_values.0.key", {1, 1, static_cast<int64_t>(past_length), 1}},
    };
    for (const auto& input : model->inputs()) {
        if (input.get_names().count("attention_mask")) {
            shapes["attention_mask"] = {1, static_cast<int64_t>(query_length + past_length)};
        }
        if (input.get_names().count("position_ids")) {
            shapes["position_ids"] = {static_cast<int64_t>(query_length)};
        }
    }
    model->reshape(shapes);
}

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& model,
                          const std::map<std::string, std::vector<int64_t>>& values = {}) {
    ov::TensorVector inputs;
    for (const auto& input : model->inputs()) {
        ov::Tensor tensor(input.get_element_type(), input.get_shape());
        std::memset(tensor.data(), 0, tensor.get_byte_size());
        const auto it = values.find(input.get_any_name());
        if (it != values.end()) {
            OPENVINO_ASSERT(tensor.get_size() == it->second.size());
            std::copy(it->second.begin(), it->second.end(), tensor.data<int64_t>());
        }
        inputs.push_back(tensor);
    }
    ov::TensorVector outputs;
    for (const auto& output : model->outputs()) {
        outputs.emplace_back(output.get_element_type(), output.get_shape());
    }
    OPENVINO_ASSERT(model->evaluate(outputs, inputs), "Qwen3-ASR reference evaluation failed");
    return outputs;
}

size_t count_inputs(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto inputs = model->inputs();
    return std::count_if(inputs.begin(), inputs.end(), [&](const auto& input) {
        return input.get_names().count(name) != 0;
    });
}

class PrepareQwen3ASRMaskTest : public ::testing::TestWithParam<bool> {};

TEST_F(TransformationTestsF, Qwen3ASRGenerateGraphMatchesReference) {
    model = make_decoder_mask_model(true, 2, false, false, true);
    model_ref = make_decoder_mask_model(true, 2, true, false, true);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    manager.register_pass<ov::npuw::PrepareQwen3ASRKVCacheModel>();
}

TEST_F(TransformationTestsF, Qwen3ASRPrefillGraphMatchesReference) {
    model = make_decoder_mask_model(false, 2, false, true, true);
    model_ref = make_decoder_mask_model(false, 2, true, true, true);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    manager.register_pass<ov::npuw::PrepareQwen3ASRPrefillModel>();
}

TEST_F(TransformationTestsF, Qwen3ASRResidualStateGraphMatchesReference) {
    using namespace ov::op;
    const auto variable =
        std::make_shared<util::Variable>(util::VariableInfo{ov::PartialShape{1, 2}, ov::element::f32, "encoder_state"});
    const auto initial = v0::Constant::create(ov::element::f32, ov::Shape{1, 2}, {0, 0});
    const auto read = std::make_shared<v6::ReadValue>(initial, variable);
    read->set_friendly_name("state_read");
    read->get_rt_info()["qwen_state_test"] = std::string("state");
    const auto assign = std::make_shared<v6::Assign>(read, variable);
    const auto result = std::make_shared<v0::Result>(read);
    result->set_friendly_name("state_result");
    model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::SinkVector{assign}, ov::ParameterVector{});

    const auto replacement = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2});
    replacement->set_friendly_name("state_read");
    replacement->output(0).set_names({"encoder_state"});
    replacement->get_rt_info()["qwen_state_test"] = std::string("state");
    const auto result_ref = std::make_shared<v0::Result>(replacement);
    result_ref->set_friendly_name("state_result");
    model_ref = std::make_shared<ov::Model>(ov::ResultVector{result_ref}, ov::ParameterVector{replacement});
    comparator.enable(FunctionsComparator::CONST_VALUES);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    manager.register_pass<ov::npuw::PrepareQwen3ASRModel>();
}

TEST_P(PrepareQwen3ASRMaskTest, StaticGenerationMasksUnusedSlotsAndKeepsAbsolutePositions) {
    constexpr size_t capacity = 1024;
    auto original = make_decoder_mask_model(GetParam());
    make_static(original, 1, capacity - 1);
    const auto unmasked = evaluate(original);
    const auto* original_weights = unmasked[0].data<const float>();
    EXPECT_GT(std::accumulate(original_weights + 53, original_weights + capacity - 1, 0.0f), 0.94f);
    EXPECT_EQ(unmasked[1].data<const int64_t>()[0], capacity - 1);

    auto model = make_decoder_mask_model(GetParam());
    ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model);
    make_static(model, 1, capacity - 1);
    for (const size_t stored : {1u, 53u, 54u, 1023u}) {
        std::vector<int64_t> mask(capacity, 1);
        std::fill_n(mask.begin(), stored, 0);
        mask.back() = 0;
        const auto outputs = evaluate(model, {{"attention_mask", mask}, {"position_ids", {int64_t(stored)}}});
        const auto* weights = outputs[0].data<const float>();
        EXPECT_FLOAT_EQ(std::accumulate(weights + stored, weights + capacity - 1, 0.0f), 0.0f);
        EXPECT_NEAR(std::accumulate(weights, weights + capacity, 0.0f), 1.0f, 1.0e-5f);
        EXPECT_NEAR(weights[capacity - 1], 1.0f / (stored + 1), 1.0e-6f);
        EXPECT_EQ(outputs[1].data<const int64_t>()[0], stored);
    }
}

TEST_P(PrepareQwen3ASRMaskTest, PrefillPreservesCausalityAndExcludesLeftPadding) {
    constexpr size_t capacity = 16;
    constexpr size_t padding = 13;
    auto model = make_decoder_mask_model(GetParam());
    ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model);
    make_static(model, capacity, 0);
    std::vector<int64_t> mask(capacity, 1);
    std::fill(mask.begin() + padding, mask.end(), 0);
    std::vector<int64_t> positions(capacity, 0);
    std::iota(positions.begin() + padding, positions.end(), 0);
    const auto outputs = evaluate(model, {{"attention_mask", mask}, {"position_ids", positions}});
    const auto* weights = outputs[0].data<const float>();
    for (size_t query = padding; query < capacity; ++query) {
        for (size_t key = 0; key < capacity; ++key) {
            const float expected = key >= padding && key <= query ? 1.0f / (query - padding + 1) : 0.0f;
            EXPECT_NEAR(weights[query * capacity + key], expected, 1.0e-6f);
        }
        EXPECT_EQ(outputs[1].data<const int64_t>()[query], query - padding);
    }
}

TEST(PrepareQwen3ASRModelTest, SharesInputsAcrossMultipleAttentionAndPositionBranches) {
    for (const bool prefill : {false, true}) {
        auto model = make_decoder_mask_model(false, 2);
        if (prefill) {
            ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model);
        } else {
            ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model);
        }
        EXPECT_EQ(count_inputs(model, "attention_mask"), 1u);
        EXPECT_EQ(count_inputs(model, "position_ids"), 1u);
    }
}

TEST(PrepareQwen3ASRModelTest, SecondRunIsANoOp) {
    for (const bool prefill : {false, true}) {
        SCOPED_TRACE(prefill);
        auto model = make_decoder_mask_model(false, 2);
        const auto run = [&] {
            return prefill ? ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model)
                           : ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model);
        };
        ASSERT_TRUE(run());
        const auto expected = model->clone();
        ASSERT_FALSE(run());
        auto comparator = FunctionsComparator::no_default();
        comparator.enable(FunctionsComparator::NODES);
        comparator.enable(FunctionsComparator::PRECISIONS);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::RUNTIME_KEYS);
        comparator.enable(FunctionsComparator::NAMES);
        comparator.enable(FunctionsComparator::TENSOR_NAMES);
        const auto comparison = comparator.compare(model, expected);
        EXPECT_TRUE(comparison.valid) << comparison.message;
    }
}

TEST(PrepareQwen3ASRModelTest, RejectsExistingPositiveAttendMaskWithoutChangingItsValues) {
    using namespace ov::op;
    for (const bool prefill : {false, true}) {
        SCOPED_TRACE(prefill);
        auto model = make_decoder_mask_model();
        auto mask = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
        mask->output(0).set_names({"attention_mask"});
        auto position = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
        position->output(0).set_names({"position_ids"});
        auto positive_is_attended = std::make_shared<v0::Convert>(mask, ov::element::boolean);
        model->add_parameters({mask, position});
        model->add_results(
            {std::make_shared<v0::Result>(positive_is_attended), std::make_shared<v0::Result>(position)});
        make_static(model, 1, 1);
        const auto before = evaluate(model, {{"attention_mask", {0, 1}}, {"position_ids", {7}}});
        EXPECT_FALSE(before[2].data<const char>()[0]);
        EXPECT_TRUE(before[2].data<const char>()[1]);
        if (prefill) {
            EXPECT_THROW(ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model), ov::Exception);
        } else {
            EXPECT_THROW(ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model), ov::Exception);
        }
        const auto after = evaluate(model, {{"attention_mask", {0, 1}}, {"position_ids", {7}}});
        EXPECT_EQ(std::memcmp(before[2].data(), after[2].data(), before[2].get_byte_size()), 0);
        EXPECT_EQ(count_inputs(model, "attention_mask"), 1u);
        EXPECT_EQ(count_inputs(model, "position_ids"), 1u);
    }
}

TEST(PrepareQwen3ASRModelTest, RejectsChangedPreparedMaskSemantics) {
    using namespace ov::op;
    auto model = make_decoder_mask_model();
    ASSERT_TRUE(ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model));
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto equal = ov::as_type_ptr<v1::Equal>(node)) {
            equal->input(1).replace_source_output(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
            break;
        }
    }
    EXPECT_THROW(ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model), ov::Exception);
}

TEST(PrepareQwen3ASRModelTest, PreservesMaskAndPositionMetadata) {
    using namespace ov::op;
    auto model = make_decoder_mask_model(false, 1, false, true, true);
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto causal = ov::as_type_ptr<v1::LessEqual>(node)) {
            causal->get_rt_info()["qwen_mask_test"] = std::string("mask");
            causal->output(0).set_names({"causal_tensor"});
        } else if (const auto reshape = ov::as_type_ptr<v1::Reshape>(node)) {
            reshape->get_rt_info()["qwen_position_test"] = std::string("position");
            const auto shape = reshape->get_input_node_shared_ptr(1);
            shape->set_friendly_name("position_shape");
            shape->get_rt_info()["qwen_shape_test"] = std::string("shape");
        }
    }
    ASSERT_TRUE(ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model));
    EXPECT_EQ(model->input("attention_mask").get_node()->get_rt_info().at("qwen_mask_test").as<std::string>(), "mask");
    EXPECT_EQ(model->input("position_ids").get_node()->get_rt_info().at("qwen_position_test").as<std::string>(),
              "position");
    bool found_mask = false;
    bool found_shape = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "causal_0" && ov::is_type<v1::LogicalAnd>(node)) {
            EXPECT_EQ(node->get_rt_info().at("qwen_mask_test").as<std::string>(), "mask");
            EXPECT_EQ(node->output(0).get_names().count("causal_tensor"), 1u);
            found_mask = true;
        }
        if (node->get_friendly_name() == "position_shape") {
            EXPECT_EQ(node->get_rt_info().at("qwen_shape_test").as<std::string>(), "shape");
            EXPECT_EQ(ov::as_type_ptr<v0::Constant>(node)->cast_vector<int64_t>().back(), -1);
            found_shape = true;
        }
    }
    EXPECT_TRUE(found_mask);
    EXPECT_TRUE(found_shape);
}

TEST(PrepareQwen3ASRModelTest, PreservesResidualStateMetadataAndIsIdempotent) {
    using namespace ov::op;
    for (const bool reuse : {false, true}) {
        SCOPED_TRACE(reuse);
        const auto variable = std::make_shared<util::Variable>(
            util::VariableInfo{ov::PartialShape{1, 2}, ov::element::f32, "encoder_state"});
        auto parameter = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2});
        parameter->set_friendly_name("encoder_input");
        parameter->output(0).set_names({"encoder_input"});
        const ov::Output<ov::Node> initial =
            reuse ? parameter->output(0) : v0::Constant::create(ov::element::f32, ov::Shape{1, 2}, {0, 0})->output(0);
        auto read = std::make_shared<v6::ReadValue>(initial, variable);
        read->set_friendly_name("state_read");
        read->output(0).set_names({"state_output"});
        read->get_rt_info()["qwen_state_test"] = std::string("state");
        auto assign = std::make_shared<v6::Assign>(read, variable);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<v0::Result>(read)},
                                                 ov::SinkVector{assign},
                                                 reuse ? ov::ParameterVector{parameter} : ov::ParameterVector{});
        ASSERT_TRUE(ov::npuw::PrepareQwen3ASRModel().run_on_model(model));
        ASSERT_EQ(model->get_parameters().size(), 1u);
        const auto replacement = model->get_parameters()[0];
        EXPECT_EQ(replacement->get_rt_info().at("qwen_state_test").as<std::string>(), "state");
        EXPECT_EQ(replacement->output(0).get_names().count("state_output"), 1u);
        EXPECT_EQ(replacement->get_friendly_name(), reuse ? "encoder_input" : "state_read");
        EXPECT_TRUE(model->get_sinks().empty());
        EXPECT_TRUE(model->get_variables().empty());
        EXPECT_FALSE(ov::npuw::PrepareQwen3ASRModel().run_on_model(model));
    }
}

TEST(PrepareQwen3ASRModelTest, CanonicalKVShapesStayIndependentOfEncoderLength) {
    constexpr uint32_t capacity = 128;
    constexpr uint32_t encoder_length = 64;
    const ov::npuw::KVAxesPosition axes{0, 2};
    for (const bool prefill : {true, false}) {
        SCOPED_TRACE(prefill);
        auto model = make_decoder_mask_model();
        if (prefill) {
            ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model);
        } else {
            ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model);
        }
        const uint32_t input_size = prefill ? capacity : 1;
        ov::npuw::ReshapeToStatic(input_size, capacity, axes, 0, encoder_length, prefill, false).run_on_model(model);

        EXPECT_EQ(model->input("past_key_values.0.key").get_shape(), (ov::Shape{1, 1, capacity - input_size, 1}));
        EXPECT_EQ(model->input("attention_mask").get_shape(), (ov::Shape{1, capacity}));
        EXPECT_EQ(model->input("position_ids").get_shape(), (ov::Shape{input_size}));
        EXPECT_EQ(model->output(0).get_shape(), (ov::Shape{1, 1, input_size, capacity}));
    }
}

TEST(PrepareQwen3ASRModelTest, WhisperCacheShapesStillUseEncoderLength) {
    constexpr uint32_t capacity = 128;
    constexpr uint32_t encoder_length = 64;
    const ov::npuw::KVAxesPosition axes{0, 2};
    for (const bool prefill : {true, false}) {
        SCOPED_TRACE(prefill);
        const uint32_t input_size = prefill ? capacity : 1;
        auto encoder_cache = make_decoder_mask_model();
        ov::npuw::ReshapeToStatic(input_size, capacity, axes, 0, encoder_length, prefill, true)
            .run_on_model(encoder_cache);
        EXPECT_EQ(encoder_cache->input("past_key_values.0.key").get_shape(), (ov::Shape{1, 1, encoder_length, 1}));

        auto decoder_cache = make_decoder_mask_model();
        decoder_cache->input("past_key_values.0.key").get_tensor().set_names({"past_key_values.0.decoder.key"});
        ov::npuw::ReshapeToStatic(input_size, capacity, axes, 0, encoder_length, prefill, true)
            .run_on_model(decoder_cache);
        EXPECT_EQ(decoder_cache->input("past_key_values.0.decoder.key").get_shape(),
                  (ov::Shape{1, 1, capacity - input_size, 1}));
    }
}

class Qwen3ASRInferRequestTest : public ::testing::TestWithParam<bool> {};

TEST_P(Qwen3ASRInferRequestTest, ResetRestartsSingleTokenPrefillAndPublishesState) {
    auto model = make_stateful_qwen_decoder();
    auto plugin = std::make_shared<ov::test::npuw::NullPlugin>();
    ov::test::npuw::RecordingFactory recorder;
    ov::AnyMap properties{{"NPUW_LLM", "YES"},
                          {"NPUW_QWEN3_ASR", "YES"},
                          {"NPUW_QWEN3_ASR_MAX_ENCODER_LEN", "16"},
                          {"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                          {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                          {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                          {"NPUW_LLM_SHARED_HEAD", GetParam() ? "YES" : "NO"},
                          {"NPUW_DEVICES", "CPU"}};
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(model, plugin, properties, recorder.make_factory());
    EXPECT_EQ(recorder.count_suffix("_lm_head"), GetParam() ? 1u : 0u);
    const auto* prefill = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill, nullptr);
    if (!GetParam()) {
        EXPECT_EQ(prefill->props.at("NPUW_SLICE_OUT").as<std::string>(), "NO");
        EXPECT_EQ(prefill->model->output("logits").get_shape().at(1), 1u);
    }
    EXPECT_TRUE(compiled->get_property("NPUW_QWEN3_ASR").as<bool>());
    EXPECT_EQ(compiled->get_property("NPUW_QWEN3_ASR_MAX_ENCODER_LEN").as<uint64_t>(), 16u);
    ov::npuw::Qwen3ASRInferRequest request(compiled);

    const auto set_input =
        [](ov::npuw::Qwen3ASRInferRequest& target, const std::string& name, const ov::Tensor& tensor) {
            const auto port = ov::npuw::util::find_port_by_name(target.get_inputs(), name);
            ASSERT_TRUE(port.has_value()) << "Missing input " << name;
            target.set_tensor(*port, ov::get_tensor_impl(tensor));
        };
    ov::Tensor encoder_hidden_states(ov::element::f32, {1, 1, 64});
    std::fill_n(encoder_hidden_states.data<float>(), encoder_hidden_states.get_size(), 1.0f);
    ov::Tensor one_token(ov::element::i64, {1, 1});
    one_token.data<int64_t>()[0] = 1;
    ov::npuw::Qwen3ASRInferRequest fresh(compiled);
    set_input(fresh, "encoder_hidden_states", encoder_hidden_states);
    set_input(fresh, "input_ids", one_token);
    ASSERT_NO_THROW(fresh.infer());
    const auto fresh_logits_port = ov::npuw::util::find_port_by_name(fresh.get_outputs(), "logits");
    ASSERT_TRUE(fresh_logits_port.has_value());
    auto fresh_logits = fresh.get_tensor(*fresh_logits_port);
    ov::Tensor expected_logits(fresh_logits->get_element_type(), fresh_logits->get_shape());
    std::memcpy(expected_logits.data(), fresh_logits->data(), fresh_logits->get_byte_size());

    set_input(request, "encoder_hidden_states", encoder_hidden_states);
    ov::Tensor prompt(ov::element::i64, {1, 2});
    std::fill_n(prompt.data<int64_t>(), prompt.get_size(), 1);
    set_input(request, "input_ids", prompt);
    ASSERT_NO_THROW(request.infer());

    const auto states = request.query_state();
    const auto state = std::find_if(states.begin(), states.end(), [](const auto& candidate) {
        return candidate->get_name() == ov::npuw::StoredTokensState::state_name;
    });
    ASSERT_NE(state, states.end());
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 2);

    set_input(request, "input_ids", one_token);
    ASSERT_NO_THROW(request.infer());
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 3);

    ASSERT_NO_THROW(fresh.infer());
    const auto fresh_states = fresh.query_state();
    const auto fresh_state = std::find_if(fresh_states.begin(), fresh_states.end(), [](const auto& candidate) {
        return candidate->get_name() == ov::npuw::StoredTokensState::state_name;
    });
    ASSERT_NE(fresh_state, fresh_states.end());
    EXPECT_EQ((*fresh_state)->get_state()->data<int64_t>()[0], 2);
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 3);

    (*state)->reset();
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 0);
    ASSERT_NO_THROW(request.infer());
    EXPECT_EQ((*state)->get_state()->data<int64_t>()[0], 1);
    const auto logits_port = ov::npuw::util::find_port_by_name(request.get_outputs(), "logits");
    ASSERT_TRUE(logits_port.has_value());
    const auto actual_logits = request.get_tensor(*logits_port);
    ASSERT_EQ(actual_logits->get_shape(), expected_logits.get_shape());
    ASSERT_EQ(actual_logits->get_byte_size(), expected_logits.get_byte_size());
    EXPECT_EQ(std::memcmp(actual_logits->data(), expected_logits.data(), expected_logits.get_byte_size()), 0);
}

TEST(Qwen3ASRPropertiesTest, ValidatesEncoderCapacityAndUsesDefault) {
    auto plugin = std::make_shared<ov::test::npuw::NullPlugin>();
    ov::AnyMap properties{{"NPUW_LLM", "YES"},
                          {"NPUW_QWEN3_ASR", "YES"},
                          {"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                          {"NPUW_LLM_MIN_RESPONSE_LEN", "64"},
                          {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                          {"NPUW_DEVICES", "CPU"}};
    for (const auto& capacity :
         {std::string{"0"}, std::to_string(static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1)}) {
        SCOPED_TRACE(capacity);
        auto invalid = properties;
        invalid["NPUW_QWEN3_ASR_MAX_ENCODER_LEN"] = capacity;
        ov::test::npuw::RecordingFactory recorder;
        EXPECT_THROW((void)std::make_shared<ov::npuw::LLMCompiledModel>(make_stateful_qwen_decoder(),
                                                                        plugin,
                                                                        invalid,
                                                                        recorder.make_factory()),
                     ov::Exception);
    }

    ov::test::npuw::RecordingFactory recorder;
    auto compiled = std::make_shared<ov::npuw::LLMCompiledModel>(make_stateful_qwen_decoder(),
                                                                 plugin,
                                                                 properties,
                                                                 recorder.make_factory());
    EXPECT_EQ(compiled->get_property("NPUW_QWEN3_ASR_MAX_ENCODER_LEN").as<uint64_t>(), 64u);
    const auto* prefill = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill, nullptr);
    EXPECT_EQ(prefill->model->input("encoder_hidden_states").get_shape(), (ov::Shape{1, 64, 64}));
}

TEST(Qwen3ASRInferRequestTest, SharedHeadMatchesUnsharedPrefillAndDecode) {
    ov::Core core;
    const auto model = make_stateful_qwen_decoder();
    ov::AnyMap properties{{"NPU_USE_NPUW", "YES"},
                          {"NPUW_LLM", "YES"},
                          {"NPUW_QWEN3_ASR", "YES"},
                          {"NPUW_QWEN3_ASR_MAX_ENCODER_LEN", "16"},
                          {"NPUW_LLM_MAX_PROMPT_LEN", "16"},
                          {"NPUW_LLM_MIN_RESPONSE_LEN", "16"},
                          {"NPUW_LLM_PREFILL_HINT", "STATIC"},
                          {"NPUW_LLM_GENERATE_PYRAMID", "NO"},
                          {"NPUW_ONLINE_PIPELINE", "NONE"},
                          {"NPUW_DEVICES", "CPU"}};
    std::vector<ov::InferRequest> requests;
    for (const bool shared_head : {true, false}) {
        SCOPED_TRACE(shared_head);
        properties["NPUW_LLM_SHARED_HEAD"] = shared_head ? "YES" : "NO";
        ov::CompiledModel compiled;
        ASSERT_NO_THROW(compiled = core.compile_model(model, "NPU", properties));
        EXPECT_EQ(compiled.get_property("NPUW_LLM_SHARED_HEAD").as<bool>(), shared_head);
        ASSERT_NO_THROW(requests.push_back(compiled.create_infer_request()));
    }

    ov::Tensor encoder{ov::element::f32, {1, 1, 64}};
    std::fill_n(encoder.data<float>(), encoder.get_size(), 0.25f);
    ov::Tensor beam{ov::element::i32, {1}};
    beam.data<int32_t>()[0] = 0;
    std::vector<float> previous_logits;
    bool changed = false;
    for (const std::vector<int64_t>& tokens :
         {std::vector<int64_t>{1, 2, 3}, std::vector<int64_t>{4}, std::vector<int64_t>{5}}) {
        SCOPED_TRACE(tokens.size());
        ov::Tensor ids{ov::element::i64, {1, tokens.size()}};
        std::copy(tokens.begin(), tokens.end(), ids.data<int64_t>());
        for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
            SCOPED_TRACE(request_index);
            auto& request = requests[request_index];
            request.set_tensor("input_ids", ids);
            request.set_tensor("beam_idx", beam);
            request.set_tensor("encoder_hidden_states", encoder);
            ASSERT_NO_THROW(request.infer());
        }
        const auto expected = requests[0].get_tensor("logits");
        const auto actual = requests[1].get_tensor("logits");
        ASSERT_EQ(actual.get_shape(), expected.get_shape());
        ASSERT_EQ(actual.get_element_type(), ov::element::f32);
        for (size_t index = 0; index < actual.get_size(); ++index) {
            ASSERT_TRUE(std::isfinite(actual.data<const float>()[index]));
            EXPECT_NEAR(actual.data<const float>()[index], expected.data<const float>()[index], 1e-5f);
            if (!previous_logits.empty() && actual.data<const float>()[index] != previous_logits[index]) {
                changed = true;
            }
        }
        previous_logits.assign(actual.data<const float>(), actual.data<const float>() + actual.get_size());
    }
    EXPECT_TRUE(changed);
}

TEST(Qwen3ASRPropertiesTest, CompiledAndImportedModelExposeOptions) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() / ("npuw_qwen_properties_" + std::to_string(nonce));
    ASSERT_TRUE(std::filesystem::create_directory(dir));
    struct Cleanup {
        std::filesystem::path dir;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(dir / "model.xml", ec);
            std::filesystem::remove(dir / "model.bin", ec);
            std::filesystem::remove(dir, ec);
        }
    } cleanup{dir};
    const auto xml = dir / "model.xml";
    const auto bin = dir / "model.bin";
    ov::Core core;
    ov::serialize(make_stateful_qwen_decoder(), xml.string(), bin.string());
    const auto model = core.read_model(xml.string(), bin.string());
    const ov::AnyMap properties{{"NPU_USE_NPUW", "YES"},
                                {"NPUW_LLM", "YES"},
                                {"NPUW_QWEN3_ASR", "YES"},
                                {"NPUW_QWEN3_ASR_MAX_ENCODER_LEN", "16"},
                                {"NPUW_DEVICES", "CPU"},
                                {"NPUW_ONLINE_PIPELINE", "NONE"},
                                {"NPUW_LLM_MAX_PROMPT_LEN", "32"},
                                {"NPUW_LLM_MIN_RESPONSE_LEN", "8"}};
    auto compiled = core.compile_model(model, "NPU", properties);
    EXPECT_TRUE(compiled.get_property("NPUW_QWEN3_ASR").as<bool>());
    EXPECT_EQ(compiled.get_property("NPUW_QWEN3_ASR_MAX_ENCODER_LEN").as<uint64_t>(), 16u);

    std::stringstream blob;
    compiled.export_model(blob);
    auto import_properties = properties;
    import_properties.erase("NPUW_QWEN3_ASR");
    import_properties.erase("NPUW_QWEN3_ASR_MAX_ENCODER_LEN");
    import_properties[ov::weights_path.name()] = bin.string();
    auto imported = core.import_model(blob, "NPU", import_properties);
    EXPECT_TRUE(imported.get_property("NPUW_QWEN3_ASR").as<bool>());
    EXPECT_EQ(imported.get_property("NPUW_QWEN3_ASR_MAX_ENCODER_LEN").as<uint64_t>(), 16u);
}

TEST(PrepareQwen3ASRModelTest, RejectsUnsupportedExportInsteadOfSilentlyFreezingAttention) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    input->output(0).set_names({"input_ids"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{input}, ov::ParameterVector{input});
    EXPECT_THROW(ov::npuw::PrepareQwen3ASRKVCacheModel().run_on_model(model), ov::Exception);
    EXPECT_THROW(ov::npuw::PrepareQwen3ASRPrefillModel().run_on_model(model), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(ExportedMaskTypes, PrepareQwen3ASRMaskTest, ::testing::Bool());
INSTANTIATE_TEST_SUITE_P(SharedHead, Qwen3ASRInferRequestTest, ::testing::Bool());

}  // namespace
