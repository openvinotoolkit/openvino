// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/optimize_lincache_layout.hpp"

#include <gtest/gtest.h>

#include <map>
#include <regex>
#include <string>
#include <string_view>

#include "llm_pass_test_fixture.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

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

// Applying the pass twice must not flip the layout back: the second run sees a
// rank-3 state again and transposes it once more, which is the expected and
// well-defined behaviour; here we only check it stays consistent and valid.
TEST_F(OptimizeLinCacheLayoutPassTest, PastAndPresentStayConsistentAfterRepeatedRuns) {
    auto model = make_stateless(ov::test::npuw::build_lfm2_llm_test_model());
    ASSERT_TRUE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(model));
    ASSERT_TRUE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(model));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    const auto past = shapes_by_name(model->inputs(), "cache_params.past.conv");
    const auto present = shapes_by_name(model->outputs(), "cache_params.present.conv");
    for (const auto& [name, shape] : present) {
        const auto past_name = std::regex_replace(name, std::regex("present"), "past");
        EXPECT_EQ(shape, past.at(past_name)) << name;
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
