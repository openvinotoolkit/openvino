// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/optimize_value_tensors.hpp"

#include <gtest/gtest.h>

#include <tuple>

#include "embedding/redirect_new_kv_to_output.hpp"
#include "llm_pass_test_fixture.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

class OptimizeValueTensorsPassTest : public ov::test::npuw::LLMPassTestFixture {
protected:
    static bool any_matmul_has_transpose_b(const std::shared_ptr<ov::Model>& model) {
        for (const auto& op : model->get_ops()) {
            if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(op)) {
                if (matmul->get_transpose_b()) {
                    return true;
                }
            }
        }
        return false;
    }
};

// ScaledDotProductAttentionDecomposition replaces every SDPA node in both the
// prefill and generate sub-models when OptimizeValueTensors is enabled.
// Parametrize over sub-model suffix to avoid duplicating the identical test body.
class OptimizeValueTensorsNoSDPATest : public ov::test::npuw::LLMPassTestFixture,
                                       public ::testing::WithParamInterface<std::string> {};

INSTANTIATE_TEST_SUITE_P(SubModels,
                         OptimizeValueTensorsNoSDPATest,
                         ::testing::Values(std::string{"_prefill"}, std::string{"_kv"}),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             auto name = info.param;
                             if (!name.empty() && name[0] == '_')
                                 name = name.substr(1);
                             return name;
                         });

TEST_P(OptimizeValueTensorsNoSDPATest, NoSDPAOpsAfterOptimization) {
    const auto& fragment = GetParam();
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& sub = require_sub_model_containing(recorder, fragment);

    EXPECT_EQ(count_ops<ov::op::v13::ScaledDotProductAttention>(sub.model), 0u);
}

// After SDPA decomposition each layer produces two MatMul ops (QK and SV multiplications).
// With 2 layers the generate model must contain at least 4 MatMul ops.
TEST_F(OptimizeValueTensorsPassTest, GenerateModelHasMatMulFromDecomposedSDPA) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    // 2 layers x 2 MatMuls per SDPA decomposition = at least 4
    EXPECT_GE(count_ops<ov::op::v0::MatMul>(generate.model), 4u);
}

// SDPA decomposition inserts a Softmax node for each layer.
// With 2 layers the generate model must contain at least 2 Softmax ops.
TEST_F(OptimizeValueTensorsPassTest, GenerateModelHasSoftmaxFromDecomposedSDPA) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    // At least one Softmax per attention layer; SDPA decomposition creates v8::Softmax
    EXPECT_GE(count_ops<ov::op::v8::Softmax>(generate.model), 2u);
}

// When OptimizeValueTensors is disabled the ScaledDotProductAttentionDecomposition
// sub-pass is skipped.  DecomposeGQA only targets GroupQueryAttention, so the
// standard test model's SDPA nodes must remain intact.
TEST_F(OptimizeValueTensorsPassTest, SDPAOpsRemainsWhenOptimizationDisabled) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "NO"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_GT(count_ops<ov::op::v13::ScaledDotProductAttention>(generate.model), 0u);
}

// TransposeValueTensors sets transpose_b=true on the value-multiplication MatMul
// when the optimisation fires.  At least one MatMul in the generate model must
// have transpose_b set after the pass runs.
TEST_F(OptimizeValueTensorsPassTest, AtLeastOneMatMulHasTransposeBSet) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model({{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}}, recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(any_matmul_has_transpose_b(generate.model));
}

// Same as above but on a GQA model (num_kv_heads < num_heads), which exercises
// the TransposeValueTensors_GQA pattern instead of the MHA one.
TEST_F(OptimizeValueTensorsPassTest, AtLeastOneMatMulHasTransposeBSet_GQA) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;

    ASSERT_NO_THROW(compiled = create_compiled_model(ov::test::npuw::build_llm_gqa_test_model(),
                                                     {{"NPUW_LLM_OPTIMIZE_V_TENSORS", "YES"}},
                                                     recorder));
    ASSERT_NE(compiled, nullptr);

    const auto& generate = require_sub_model_containing(recorder, "_kv");

    EXPECT_TRUE(any_matmul_has_transpose_b(generate.model));
}

TEST_F(OptimizeValueTensorsPassTest, DirectValuePatternsDoNotDependOnNodeNames) {
    auto generate_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    auto generate_scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 5, 3});
    auto generate_softmax = std::make_shared<ov::op::v8::Softmax>(generate_scores);
    auto generate_matmul = std::make_shared<ov::op::v0::MatMul>(generate_softmax, generate_value);
    auto generate_model = std::make_shared<ov::Model>(ov::OutputVector{generate_matmul},
                                                      ov::ParameterVector{generate_value, generate_scores});

    EXPECT_TRUE(ov::npuw::util::OptimizeValueTensors(false, true).run_on_model(generate_model));
    EXPECT_TRUE(generate_matmul->get_transpose_b());
    EXPECT_EQ(generate_value->get_partial_shape(), ov::PartialShape({1, 2, 4, 3}));

    auto prefill_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 2, 4});
    auto prefill_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    auto prefill_transpose = std::make_shared<ov::op::v1::Transpose>(prefill_value, prefill_order);
    auto prefill_scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 5, 3});
    auto prefill_softmax = std::make_shared<ov::op::v8::Softmax>(prefill_scores);
    auto prefill_matmul = std::make_shared<ov::op::v0::MatMul>(prefill_softmax, prefill_transpose);
    auto prefill_model = std::make_shared<ov::Model>(ov::OutputVector{prefill_matmul},
                                                     ov::ParameterVector{prefill_value, prefill_scores});

    EXPECT_TRUE(ov::npuw::util::OptimizeValueTensors(true, true).run_on_model(prefill_model));
    EXPECT_TRUE(prefill_matmul->get_transpose_b());
    EXPECT_EQ(prefill_transpose->get_output_partial_shape(0), ov::PartialShape({1, 2, 4, 3}));
}

TEST_F(OptimizeValueTensorsPassTest, SharedDirectGenerateValueIsTransposedOnce) {
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    auto scores_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 5, 3});
    auto scores_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 6, 3});
    auto softmax_0 = std::make_shared<ov::op::v8::Softmax>(scores_0);
    auto softmax_1 = std::make_shared<ov::op::v8::Softmax>(scores_1);
    auto matmul_0 = std::make_shared<ov::op::v0::MatMul>(softmax_0, value);
    auto matmul_1 = std::make_shared<ov::op::v0::MatMul>(softmax_1, value);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{matmul_0, matmul_1},
                                             ov::ParameterVector{value, scores_0, scores_1});

    ASSERT_NO_THROW(ov::npuw::util::OptimizeValueTensors(false, true).run_on_model(model));
    EXPECT_TRUE(matmul_0->get_transpose_b());
    EXPECT_TRUE(matmul_1->get_transpose_b());
    EXPECT_EQ(value->get_partial_shape(), ov::PartialShape({1, 2, 4, 3}));
}

TEST_F(OptimizeValueTensorsPassTest, SharedConcatGenerateValueIsTransposedOnce) {
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3, 4});
    ov::OutputVector results;
    ov::ParameterVector parameters{value};
    std::vector<std::shared_ptr<ov::op::v0::Concat>> concats;
    std::vector<std::shared_ptr<ov::op::v0::MatMul>> matmuls;

    for (std::size_t query_length : {5u, 6u}) {
        auto new_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 2, 4});
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(new_value, order);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{value, transpose}, 2);
        auto scores =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                    ov::PartialShape{1, 2, static_cast<int64_t>(query_length), 4});
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(softmax, concat);

        results.push_back(matmul);
        parameters.push_back(new_value);
        parameters.push_back(scores);
        concats.push_back(concat);
        matmuls.push_back(matmul);
    }

    auto model = std::make_shared<ov::Model>(results, parameters);
    ASSERT_NO_THROW(ov::npuw::util::OptimizeValueTensors(false).run_on_model(model));
    EXPECT_EQ(value->get_partial_shape(), ov::PartialShape({1, 2, 4, 3}));
    for (std::size_t index = 0; index < matmuls.size(); ++index) {
        EXPECT_EQ(concats[index]->get_axis(), 3u);
        EXPECT_TRUE(matmuls[index]->get_transpose_b());
    }
}

TEST_F(OptimizeValueTensorsPassTest, SharedGqaConcatGenerateValueIsTransposedOnce) {
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 3, 4});
    ov::OutputVector results;
    ov::ParameterVector parameters{value};
    std::vector<std::shared_ptr<ov::op::v0::Concat>> concats;
    std::vector<std::shared_ptr<ov::op::v3::Broadcast>> broadcasts;
    std::vector<std::shared_ptr<ov::op::v1::Reshape>> reshapes;
    std::vector<std::shared_ptr<ov::op::v0::MatMul>> matmuls;

    for (std::size_t query_length : {5u, 6u}) {
        auto new_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 1, 4});
        auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(new_value, transpose_order);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{value, transpose}, 2);
        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(concat, unsqueeze_axes);
        auto broadcast_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {1, 1, 2, 5, 4});
        auto broadcast =
            std::make_shared<ov::op::v3::Broadcast>(unsqueeze, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 2, 5, 4});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(broadcast, reshape_shape, false);
        auto scores =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                    ov::PartialShape{1, 2, static_cast<int64_t>(query_length), 5});
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(softmax, reshape);

        results.push_back(matmul);
        parameters.push_back(new_value);
        parameters.push_back(scores);
        concats.push_back(concat);
        broadcasts.push_back(broadcast);
        reshapes.push_back(reshape);
        matmuls.push_back(matmul);
    }

    auto model = std::make_shared<ov::Model>(results, parameters);
    ASSERT_NO_THROW(ov::npuw::util::OptimizeValueTensors(false).run_on_model(model));
    EXPECT_EQ(value->get_partial_shape(), ov::PartialShape({1, 1, 4, 3}));
    for (std::size_t index = 0; index < matmuls.size(); ++index) {
        EXPECT_EQ(concats[index]->get_axis(), 3u);
        EXPECT_EQ(broadcasts[index]->get_output_partial_shape(0), ov::PartialShape({1, 1, 2, 4, 5}));
        EXPECT_EQ(reshapes[index]->get_output_partial_shape(0), ov::PartialShape({1, 2, 4, 5}));
        EXPECT_TRUE(matmuls[index]->get_transpose_b());
    }
}

class OptimizeValueTensorsCacheLayoutTest : public ov::test::npuw::LLMPassTestFixture,
                                            public ::testing::WithParamInterface<std::tuple<bool, bool, size_t>> {};

TEST_P(OptimizeValueTensorsCacheLayoutTest, ShapeCorrectionLeavesDataConsumersConnected) {
    const auto [gqa, prefill, past_length] = GetParam();
    const size_t heads = 2;
    const size_t query_heads = gqa ? 4 : heads;
    const size_t head_size = 4;
    const size_t new_length = prefill ? 3 : 1;
    const size_t total_length = past_length + new_length;
    auto past = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, heads, past_length, head_size});
    auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, new_length, heads, head_size});
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(value, order);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, transpose}, 2);
    ov::Output<ov::Node> attention_value = concat;
    ov::OutputVector shape_outputs;
    std::vector<ov::Shape> original_shapes;
    if (gqa) {
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(concat, axes);
        auto broadcast_shape =
            ov::op::v0::Constant::create(ov::element::i64,
                                         ov::Shape{5},
                                         ov::Shape{1, heads, query_heads / heads, total_length, head_size});
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(unsqueeze, broadcast_shape);
        auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64,
                                                          ov::Shape{4},
                                                          ov::Shape{1, query_heads, total_length, head_size});
        attention_value = std::make_shared<ov::op::v1::Reshape>(broadcast, reshape_shape, false);
        for (const auto& output : ov::OutputVector{unsqueeze, broadcast, attention_value}) {
            original_shapes.push_back(output.get_shape());
            shape_outputs.push_back(std::make_shared<ov::op::v3::ShapeOf>(output));
        }
    }
    auto scores =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, query_heads, new_length, total_length});
    auto softmax = std::make_shared<ov::op::v8::Softmax>(scores);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(softmax, attention_value);
    auto cache_result = std::make_shared<ov::op::v0::Result>(concat);
    cache_result->set_friendly_name("present.0.value");
    auto ordinary_result = std::make_shared<ov::op::v0::Result>(concat);
    ordinary_result->set_friendly_name("debug_value");
    auto external_consumer = std::make_shared<ov::op::v0::Convert>(concat, ov::element::f16);
    ov::OutputVector outputs{matmul, cache_result, ordinary_result, external_consumer};
    outputs.insert(outputs.end(), shape_outputs.begin(), shape_outputs.end());
    auto model = std::make_shared<ov::Model>(outputs, ov::ParameterVector{past, value, scores});

    ASSERT_TRUE(ov::npuw::util::OptimizeValueTensors(prefill).run_on_model(model));
    EXPECT_EQ(past->get_shape(), (ov::Shape{1, heads, head_size, past_length}));
    EXPECT_EQ(cache_result->input_value(0), concat->output(0));
    EXPECT_EQ(cache_result->get_shape(), (ov::Shape{1, heads, head_size, total_length}));
    EXPECT_EQ(ordinary_result->input_value(0), concat->output(0));
    EXPECT_EQ(external_consumer->input_value(0), concat->output(0));
    EXPECT_EQ(ordinary_result->get_shape(), (ov::Shape{1, heads, head_size, total_length}));
    EXPECT_EQ(external_consumer->get_shape(), (ov::Shape{1, heads, head_size, total_length}));
    EXPECT_EQ(count_ops<ov::op::v1::Transpose>(model), 1u);
    EXPECT_TRUE(matmul->get_transpose_b());
    EXPECT_EQ(matmul->get_shape(), (ov::Shape{1, query_heads, new_length, head_size}));
    if (gqa) {
        ov::OutputVector restored_shapes;
        ov::TensorVector shape_values;
        for (size_t i = 0; i < shape_outputs.size(); ++i) {
            auto restored_shape = model->get_results().at(4 + i)->input_value(0);
            ASSERT_TRUE(ov::is_type<ov::op::v8::Gather>(restored_shape.get_node()));
            restored_shapes.push_back(restored_shape);
            shape_values.emplace_back(ov::element::i64, ov::Shape{original_shapes[i].size()});
        }
        auto shape_model = std::make_shared<ov::Model>(restored_shapes, ov::ParameterVector{past, value});
        EXPECT_EQ(count_ops<ov::op::v1::Transpose>(shape_model), 1u);
        ASSERT_TRUE(shape_model->evaluate(shape_values,
                                          ov::TensorVector{ov::Tensor(ov::element::f32, past->get_shape()),
                                                           ov::Tensor(ov::element::f32, value->get_shape())}));
        for (size_t i = 0; i < shape_values.size(); ++i) {
            EXPECT_EQ(std::vector<int64_t>(shape_values[i].data<int64_t>(),
                                           shape_values[i].data<int64_t>() + original_shapes[i].size()),
                      std::vector<int64_t>(original_shapes[i].begin(), original_shapes[i].end()));
        }
    }

    ov::npuw::RedirectNewKvToOutput().run_on_model(model);
    EXPECT_EQ(cache_result->input_value(0), transpose->output(0));
    EXPECT_EQ(cache_result->get_shape(), (ov::Shape{1, heads, head_size, new_length}));
    EXPECT_EQ(cache_result->get_friendly_name(), "present.0.value");
    EXPECT_EQ(external_consumer->input_value(0), concat->output(0));
    EXPECT_EQ(external_consumer->get_shape(), (ov::Shape{1, heads, head_size, total_length}));
}

INSTANTIATE_TEST_SUITE_P(MhaAndGqa,
                         OptimizeValueTensorsCacheLayoutTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(size_t{0}, size_t{8})),
                         [](const ::testing::TestParamInfo<OptimizeValueTensorsCacheLayoutTest::ParamType>& info) {
                             const auto gqa = std::get<0>(info.param);
                             const auto prefill = std::get<1>(info.param);
                             const auto past_length = std::get<2>(info.param);
                             return std::string(gqa ? "GQA" : "MHA") + (prefill ? "_Prefill" : "_Generate") + "_Past" +
                                    std::to_string(past_length);
                         });

class OptimizeValueTensorsDynamicShapeTest : public ov::test::npuw::LLMPassTestFixture,
                                             public ::testing::WithParamInterface<std::tuple<bool, bool, int>> {};

TEST_P(OptimizeValueTensorsDynamicShapeTest, ShapeQueriesPreserveRuntimeDimensions) {
    const auto [prefill, with_convert, shape_version] = GetParam();
    auto value = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        prefill ? ov::PartialShape{1, ov::Dimension(1, 16), 2, ov::Dimension(1, 16)}
                : ov::PartialShape{1, 2, ov::Dimension(1, 16), ov::Dimension(1, 16)});
    ov::Output<ov::Node> attention_value = value;
    if (with_convert) {
        attention_value = std::make_shared<ov::op::v0::Convert>(attention_value, ov::element::f32);
    }
    if (prefill) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        attention_value = std::make_shared<ov::op::v1::Transpose>(attention_value, order);
    }
    std::shared_ptr<ov::Node> shape_of;
    if (shape_version == 0) {
        shape_of = std::make_shared<ov::op::v0::ShapeOf>(attention_value);
    } else {
        shape_of = std::make_shared<ov::op::v3::ShapeOf>(attention_value,
                                                         shape_version == 1 ? ov::element::i32 : ov::element::i64);
    }
    shape_of->set_friendly_name("logical_value_shape");
    shape_of->get_rt_info()["test_shape_attribute"] = std::string("preserved");
    auto shape_reference = std::make_shared<ov::Model>(ov::OutputVector{shape_of}, ov::ParameterVector{value})->clone();
    ov::OutputVector outputs;
    ov::ParameterVector parameters{value};
    for (size_t query_length : {1u, 2u}) {
        auto scores = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f32,
            ov::PartialShape{1, 2, static_cast<int64_t>(query_length), ov::Dimension(1, 16)});
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores);
        outputs.push_back(std::make_shared<ov::op::v0::MatMul>(softmax, attention_value));
        parameters.push_back(scores);
    }
    outputs.push_back(shape_of);
    auto model = std::make_shared<ov::Model>(outputs, parameters);
    ASSERT_TRUE(ov::npuw::util::OptimizeValueTensors(prefill, true).run_on_model(model));
    auto restored_shape = model->get_results().back()->input_value(0);
    ASSERT_TRUE(ov::is_type<ov::op::v8::Gather>(restored_shape.get_node()));
    EXPECT_EQ(restored_shape.get_element_type(), shape_of->get_output_element_type(0));
    EXPECT_EQ(restored_shape.get_node()->get_friendly_name(), "logical_value_shape");
    EXPECT_EQ(restored_shape.get_node()->get_rt_info().at("test_shape_attribute").as<std::string>(), "preserved");
    auto shape_model = std::make_shared<ov::Model>(ov::OutputVector{restored_shape}, ov::ParameterVector{value});
    EXPECT_EQ(count_ops<ov::op::v1::Transpose>(shape_model), prefill ? 1u : 0u);
    EXPECT_TRUE(value->get_partial_shape().is_dynamic());
    for (const auto& dimensions : {ov::Shape{3, 4}, ov::Shape{7, 9}}) {
        const auto sequence = dimensions[0];
        const auto embedding = dimensions[1];
        const auto logical = prefill ? ov::Shape{1, sequence, 2, embedding} : ov::Shape{1, 2, sequence, embedding};
        const auto physical = prefill ? logical : ov::Shape{1, 2, embedding, sequence};
        const auto type = restored_shape.get_element_type();
        ov::TensorVector expected{ov::Tensor(type, ov::Shape{4})};
        ov::TensorVector actual{ov::Tensor(type, ov::Shape{4})};
        ASSERT_TRUE(shape_reference->evaluate(expected, ov::TensorVector{ov::Tensor(ov::element::f32, logical)}));
        ASSERT_TRUE(shape_model->evaluate(actual, ov::TensorVector{ov::Tensor(ov::element::f32, physical)}));
        for (size_t i = 0; i < 4; ++i) {
            if (type == ov::element::i32) {
                EXPECT_EQ(actual[0].data<int32_t>()[i], expected[0].data<int32_t>()[i]);
            } else {
                EXPECT_EQ(actual[0].data<int64_t>()[i], expected[0].data<int64_t>()[i]);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ShapeOfVersions,
                         OptimizeValueTensorsDynamicShapeTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Values(0, 1, 2)));

}  // namespace
