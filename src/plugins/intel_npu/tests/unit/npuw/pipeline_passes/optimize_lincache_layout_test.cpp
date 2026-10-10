// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/optimize_lincache_layout.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include "llm_pass_test_fixture.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

ov::PartialShape swap_last_two(const ov::PartialShape& shape) {
    ov::PartialShape swapped = shape;
    std::swap(swapped[1], swapped[2]);
    return swapped;
}

// Collects the shapes of all inputs (outputs) whose names contain `needle`, keyed by name.
template <class Ports>
std::map<std::string, ov::PartialShape> shapes_by_name(const Ports& ports, std::string_view needle) {
    std::map<std::string, ov::PartialShape> result;
    for (const auto& port : ports) {
        for (const auto& name : port.get_names()) {
            if (name.find(needle) != std::string::npos) {
                result[name] = port.get_partial_shape();
            }
        }
    }
    return result;
}

class OptimizeLinCacheLayoutPassTest : public ov::test::npuw::LLMPassTestFixture {
protected:
    // Stateless models for direct pass testing: conv states become cache_params.{past,present}.conv.N
    static std::shared_ptr<ov::Model> make_stateless(std::shared_ptr<ov::Model> model) {
        ov::pass::StatefulToStateless().run_on_model(model);
        return model;
    }

    // The pass must swap the last two dims of every conv state pair and leave every other I/O alone.
    static void expect_conv_states_relaid_out(const std::shared_ptr<ov::Model>& reference,
                                              const std::shared_ptr<ov::Model>& optimized) {
        const auto ref_past = shapes_by_name(reference->inputs(), "cache_params.past.conv");
        const auto opt_past = shapes_by_name(optimized->inputs(), "cache_params.past.conv");
        ASSERT_FALSE(ref_past.empty());
        ASSERT_EQ(ref_past.size(), opt_past.size());
        for (const auto& [name, shape] : ref_past) {
            ASSERT_TRUE(opt_past.count(name)) << name;
            EXPECT_EQ(opt_past.at(name), swap_last_two(shape)) << name;
        }

        const auto ref_present = shapes_by_name(reference->outputs(), "cache_params.present.conv");
        const auto opt_present = shapes_by_name(optimized->outputs(), "cache_params.present.conv");
        ASSERT_EQ(ref_present.size(), ref_past.size());
        ASSERT_EQ(ref_present.size(), opt_present.size());
        for (const auto& [name, shape] : ref_present) {
            ASSERT_TRUE(opt_present.count(name)) << name;
            EXPECT_EQ(opt_present.at(name), swap_last_two(shape)) << name;
            // present and past of the same layer must stay byte-copy compatible
            const auto past_name = std::regex_replace(name, std::regex("present"), "past");
            EXPECT_EQ(opt_present.at(name), opt_past.at(past_name)) << name;
        }

        // Everything that is not a conv state keeps its shape (SSM states, KV cache, activations).
        auto non_conv = [](const auto& ports) {
            std::map<std::string, ov::PartialShape> result;
            for (const auto& port : ports) {
                const auto name = port.get_any_name();
                if (name.find(".conv.") == std::string::npos) {
                    result[name] = port.get_partial_shape();
                }
            }
            return result;
        };
        EXPECT_EQ(non_conv(reference->inputs()), non_conv(optimized->inputs()));
        EXPECT_EQ(non_conv(reference->outputs()), non_conv(optimized->outputs()));
    }
};

// Direct pass test on the GatedDeltaNet hybrid model (conv + SSM states).
TEST_F(OptimizeLinCacheLayoutPassTest, TransposesConvStatesOnly) {
    auto reference = make_stateless(ov::test::npuw::build_hybrid_llm_test_model());
    auto optimized = reference->clone();

    EXPECT_TRUE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(optimized));
    ASSERT_NO_THROW(optimized->validate_nodes_and_infer_types());

    expect_conv_states_relaid_out(reference, optimized);

    // One Transpose per Parameter plus one per Result.
    const auto n_states = shapes_by_name(reference->inputs(), "cache_params.past.conv").size();
    EXPECT_EQ(count_ops<ov::op::v1::Transpose>(optimized) - count_ops<ov::op::v1::Transpose>(reference), 2u * n_states);
}

// LFM2-style short-conv states have a different kernel size but the same [batch, channels, kernel] layout.
TEST_F(OptimizeLinCacheLayoutPassTest, TransposesShortConvStates) {
    auto reference = make_stateless(ov::test::npuw::build_lfm2_llm_test_model());
    auto optimized = reference->clone();

    EXPECT_TRUE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(optimized));
    expect_conv_states_relaid_out(reference, optimized);
}

// Models without linear-attention states are left untouched.
TEST_F(OptimizeLinCacheLayoutPassTest, NoConvStatesIsNoOp) {
    auto model = make_stateless(ov::test::npuw::build_llm_test_model());
    const auto n_ops = model->get_ops().size();

    EXPECT_FALSE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), n_ops);
}

TEST_F(OptimizeLinCacheLayoutPassTest, RepeatedRunsAreNoOpIncludingAfterClone) {
    auto model = make_stateless(ov::test::npuw::build_lfm2_llm_test_model());
    ASSERT_TRUE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(model));
    const auto past = shapes_by_name(model->inputs(), "cache_params.past.conv");
    const auto present = shapes_by_name(model->outputs(), "cache_params.present.conv");
    const auto n_ops = model->get_ops().size();

    for (const auto& candidate : {model, model->clone()}) {
        EXPECT_FALSE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(candidate));
        EXPECT_EQ(candidate->get_ops().size(), n_ops);
        EXPECT_EQ(shapes_by_name(candidate->inputs(), "cache_params.past.conv"), past);
        EXPECT_EQ(shapes_by_name(candidate->outputs(), "cache_params.present.conv"), present);
        ASSERT_NO_THROW(candidate->validate_nodes_and_infer_types());
    }
}

TEST_F(OptimizeLinCacheLayoutPassTest, PreservesPassThroughStateNames) {
    for (const bool cross_layer : {false, true}) {
        SCOPED_TRACE(cross_layer);
        ov::ParameterVector params;
        ov::ResultVector results;
        for (size_t i = 0; i < 2; ++i) {
            auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
            param->output(0).set_names({"cache_params.past.conv." + std::to_string(i),
                                        "cache_params.present.conv." + std::to_string(cross_layer ? 1 - i : i)});
            params.push_back(param);
            results.push_back(std::make_shared<ov::op::v0::Result>(param));
        }
        auto reference = std::make_shared<ov::Model>(results, params);
        auto optimized = reference->clone();
        ASSERT_TRUE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(optimized));
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_EQ(optimized->input(i).get_names(), reference->input(i).get_names());
            EXPECT_EQ(optimized->output(i).get_names(), reference->output(i).get_names());
            EXPECT_EQ(optimized->output(i).get_shape(), (ov::Shape{1, 3, 2}));
        }
    }
}

TEST_F(OptimizeLinCacheLayoutPassTest, UnsupportedPairLeavesAllStatesUnchanged) {
    for (const bool missing_present : {false, true}) {
        SCOPED_TRACE(missing_present);
        auto model = make_stateless(ov::test::npuw::build_lfm2_llm_test_model());
        const auto port = find_output(model, "cache_params.present.conv.1");
        ASSERT_TRUE(port.has_value());
        auto result = model->get_results().at(model->get_result_index(*port));
        if (missing_present) {
            model->remove_result(result);
        } else {
            const auto names = result->input_value(0).get_names();
            result->input(0).replace_source_output(
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 3}, {0}));
            result->input_value(0).set_names(names);
            model->validate_nodes_and_infer_types();
        }
        const auto past = shapes_by_name(model->inputs(), "cache_params.past.conv");
        const auto present = shapes_by_name(model->outputs(), "cache_params.present.conv");
        const auto n_ops = model->get_ops().size();
        EXPECT_FALSE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(model));
        EXPECT_EQ(model->get_ops().size(), n_ops);
        EXPECT_EQ(shapes_by_name(model->inputs(), "cache_params.past.conv"), past);
        EXPECT_EQ(shapes_by_name(model->outputs(), "cache_params.present.conv"), present);
    }
}

// Exercise the cache recurrence with evaluable ops, independently of device compilation.
std::shared_ptr<ov::Model> make_cache_step_model(int64_t token_count) {
    auto past = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
    past->output(0).set_names({"cache_params.past.conv.0"});
    auto tokens = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 2, token_count});
    tokens->output(0).set_names({"tokens"});
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, tokens}, 2);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto present = std::make_shared<ov::op::v8::Slice>(
        concat,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {token_count}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {token_count + 3}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
        axis);
    present->output(0).set_names({"cache_params.present.conv.0"});
    auto activation = std::make_shared<ov::op::v1::ReduceSum>(concat, axis, false);
    activation->output(0).set_names({"activation"});
    return std::make_shared<ov::Model>(ov::OutputVector{activation, present}, ov::ParameterVector{past, tokens});
}

TEST_F(OptimizeLinCacheLayoutPassTest, PreservesNonzeroStateAcrossPrefillAndDecode) {
    std::vector<std::shared_ptr<ov::Model>> references{make_cache_step_model(2),
                                                       make_cache_step_model(1),
                                                       make_cache_step_model(1)};
    std::vector<std::shared_ptr<ov::Model>> optimized;
    for (const auto& reference : references) {
        optimized.push_back(reference->clone());
        ASSERT_TRUE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(optimized.back()));
    }
    ov::Tensor ref_state(ov::element::f32, {1, 2, 3});
    ov::Tensor opt_state(ov::element::f32, {1, 3, 2});
    std::iota(ref_state.data<float>(), ref_state.data<float>() + ref_state.get_size(), 1.0f);
    for (size_t c = 0; c < 2; ++c) {
        for (size_t k = 0; k < 3; ++k) {
            opt_state.data<float>()[k * 2 + c] = ref_state.data<float>()[c * 3 + k];
        }
    }
    // Two prefill chunks, then decode with a switch to a separate generate model.
    size_t step = 0;
    for (const size_t variant : {0, 0, 1, 1, 2, 2}) {
        SCOPED_TRACE(step);
        const auto& reference = references.at(variant);
        const auto& transformed = optimized.at(variant);
        ov::Tensor tokens(ov::element::f32, reference->input(1).get_shape());
        std::iota(tokens.data<float>(), tokens.data<float>() + tokens.get_size(), 10.0f * (++step));
        ov::TensorVector ref_outputs, opt_outputs;
        for (size_t i = 0; i < reference->outputs().size(); ++i) {
            ref_outputs.emplace_back(ov::element::f32, reference->output(i).get_shape());
            opt_outputs.emplace_back(ov::element::f32, transformed->output(i).get_shape());
        }
        ASSERT_TRUE(reference->evaluate(ref_outputs, ov::TensorVector{ref_state, tokens}));
        ASSERT_TRUE(transformed->evaluate(opt_outputs, ov::TensorVector{opt_state, tokens}));
        for (size_t i = 0; i < ref_outputs[0].get_size(); ++i) {
            EXPECT_FLOAT_EQ(ref_outputs[0].data<float>()[i], opt_outputs[0].data<float>()[i]);
        }
        for (size_t c = 0; c < 2; ++c) {
            for (size_t k = 0; k < 3; ++k) {
                EXPECT_FLOAT_EQ(ref_outputs[1].data<float>()[c * 3 + k], opt_outputs[1].data<float>()[k * 2 + c]);
            }
        }
        ref_outputs[1].copy_to(ref_state);
        opt_outputs[1].copy_to(opt_state);
    }
}

// Pipeline-level: both the prefill and the generate sub-models get the new layout when the
// option is on, and keep the original one when it is off.
class OptimizeLinCacheLayoutPipelineTest : public OptimizeLinCacheLayoutPassTest,
                                           public ::testing::WithParamInterface<std::string> {};

INSTANTIATE_TEST_SUITE_P(SubModels,
                         OptimizeLinCacheLayoutPipelineTest,
                         ::testing::Values(std::string{"_prefill"}, std::string{"_kv"}),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                             return info.param.substr(1);
                         });

TEST_P(OptimizeLinCacheLayoutPipelineTest, ConvStatesTransposedInSubModel) {
    const auto& fragment = GetParam();

    RecordingFactory recorder_off;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled_off;
    ASSERT_NO_THROW(compiled_off = create_compiled_model(ov::test::npuw::build_lfm2_llm_test_model(),
                                                         {{"NPUW_LLM_OPTIMIZE_LINCACHE_LAYOUT", "NO"}},
                                                         recorder_off));
    ASSERT_NE(compiled_off, nullptr);

    RecordingFactory recorder_on;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled_on;
    ASSERT_NO_THROW(compiled_on = create_compiled_model(ov::test::npuw::build_lfm2_llm_test_model(),
                                                        {{"NPUW_LLM_OPTIMIZE_LINCACHE_LAYOUT", "YES"}},
                                                        recorder_on));
    ASSERT_NE(compiled_on, nullptr);

    const auto& sub_off = require_sub_model_containing(recorder_off, fragment);
    const auto& sub_on = require_sub_model_containing(recorder_on, fragment);
    expect_conv_states_relaid_out(sub_off.model, sub_on.model);
}

}  // namespace
