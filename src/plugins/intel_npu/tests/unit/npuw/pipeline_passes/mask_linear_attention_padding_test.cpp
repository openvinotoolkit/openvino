// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/mask_linear_attention_padding.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "llm_pass_test_fixture.hpp"
#include "npuw_transformations/kv_axes_position.hpp"
#include "npuw_transformations/optimize_lincache_layout.hpp"
#include "npuw_transformations/reshape_to_static.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

namespace {

using ov::test::npuw::RecordingFactory;

constexpr uint32_t kTotal = 32;  // static attention_mask length of the prefill model
constexpr uint32_t kStored = 8;  // tokens already processed by the previous chunks

class MaskLinearAttentionPaddingTest : public ov::test::npuw::LLMPassTestFixture {
protected:
    static std::shared_ptr<ov::Model> stateless(std::shared_ptr<ov::Model> model) {
        ov::pass::StatefulToStateless().run_on_model(model);
        return model;
    }

    // Static prefill model as NPUW builds it for a chunked prefill with the given chunk size.
    static std::shared_ptr<ov::Model> make_static(const std::shared_ptr<ov::Model>& stateless_model, uint32_t chunk) {
        auto model = stateless_model->clone();
        ov::npuw::ReshapeToStatic(chunk, kTotal, ov::npuw::KVAxesPosition{0u, 2u}, 0u, 0u, true).run_on_model(model);
        return model;
    }

    static std::map<std::string, ov::Tensor> run(const std::shared_ptr<ov::Model>& model,
                                                 uint32_t chunk,
                                                 uint32_t padding,
                                                 const std::map<std::string, ov::Tensor>& lincache_past) {
        ov::Core core;
        auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();

        for (const auto& input : compiled.inputs()) {
            const auto name = input.get_any_name();
            auto tensor = request.get_tensor(input);
            std::memset(tensor.data(), 0, tensor.get_byte_size());
            if (name == "input_ids") {
                auto* ids = tensor.data<int64_t>();
                // Real tokens right-aligned, `padding` pad tokens in front of them.
                for (uint32_t i = padding; i < chunk; ++i) {
                    ids[i] = 3 + 7 * (i - padding);
                }
            } else if (name == "attention_mask") {
                auto* mask = tensor.data<int64_t>();
                for (uint32_t i = 0; i < kStored; ++i) {
                    mask[i] = 1;
                }
                for (uint32_t i = kTotal - chunk + padding; i < kTotal; ++i) {
                    mask[i] = 1;
                }
            } else if (name == "position_ids") {
                auto* pos = tensor.data<int64_t>();
                for (uint32_t i = padding; i < chunk; ++i) {
                    pos[i] = kStored + i - padding;
                }
            } else if (lincache_past.count(name)) {
                lincache_past.at(name).copy_to(tensor);
            }
        }
        request.infer();

        std::map<std::string, ov::Tensor> outputs;
        for (const auto& output : compiled.outputs()) {
            const auto name = output.get_any_name();
            if (name.find("cache_params.present") != std::string::npos || name == "logits") {
                ov::Tensor copy(output.get_element_type(), output.get_shape());
                request.get_tensor(output).copy_to(copy);
                outputs[name] = copy;
            }
        }
        return outputs;
    }

    static std::map<std::string, ov::Tensor> make_lincache_past(const std::shared_ptr<ov::Model>& model) {
        std::map<std::string, ov::Tensor> past;
        size_t seed = 1;
        for (const auto& input : model->inputs()) {
            const auto name = input.get_any_name();
            if (name.find("cache_params.past") == std::string::npos) {
                continue;
            }
            ov::Tensor tensor(input.get_element_type(), input.get_shape());
            auto* data = tensor.data<float>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = 0.5f * std::sin(0.37f * static_cast<float>(i + 11 * seed));
            }
            ++seed;
            past[name] = tensor;
        }
        return past;
    }

    // Number of Loop inputs (summed over all Loops) that are fed by a Multiply, i.e. masked g / beta.
    static size_t count_masked_loop_inputs(const std::shared_ptr<ov::Model>& model) {
        size_t masked = 0;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node)) {
                for (const auto& desc : loop->get_input_descriptions()) {
                    masked += ov::is_type<ov::op::v1::Multiply>(loop->input_value(desc->m_input_index).get_node());
                }
            }
        }
        return masked;
    }

    // Feeds the gate of every recurrence Loop in f16 and converts it back in the body right before Exp,
    // the mixed-precision layout FuseGatedDeltaNet also accepts.
    static void convert_gate_before_exp(const std::shared_ptr<ov::Model>& model) {
        for (const auto& node : model->get_ordered_ops()) {
            const auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
            if (!loop) {
                continue;
            }
            const auto body = loop->get_function();
            for (const auto& desc : loop->get_input_descriptions()) {
                const auto param = body->get_parameters().at(desc->m_body_parameter_index);
                const auto targets = param->output(0).get_target_inputs();
                const bool is_gate = std::any_of(targets.begin(), targets.end(), [](const ov::Input<ov::Node>& t) {
                    return ov::is_type<ov::op::v0::Exp>(t.get_node());
                });
                if (!is_gate) {
                    continue;
                }
                const auto type = param->get_element_type();
                param->set_element_type(ov::element::f16);
                const auto back = std::make_shared<ov::op::v0::Convert>(param, type);
                for (auto target : targets) {
                    target.replace_source_output(back);
                }
                const auto source = loop->input_value(desc->m_input_index);
                loop->input(desc->m_input_index)
                    .replace_source_output(std::make_shared<ov::op::v0::Convert>(source, ov::element::f16));
            }
            body->validate_nodes_and_infer_types();
        }
        model->validate_nodes_and_infer_types();
    }

    // Applies `edit` to the body of every recurrence Loop together with its per-token g and beta parameters.
    template <class Edit>
    static void edit_loop_bodies(const std::shared_ptr<ov::Model>& model, const Edit& edit) {
        for (const auto& node : model->get_ordered_ops()) {
            const auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
            if (!loop) {
                continue;
            }
            const auto body = loop->get_function();
            std::shared_ptr<ov::op::v0::Parameter> gate;
            std::shared_ptr<ov::op::v0::Parameter> beta;
            for (const auto& desc : loop->get_input_descriptions()) {
                if (!ov::is_type<ov::op::util::SubGraphOp::SliceInputDescription>(desc) ||
                    loop->get_input_partial_shape(desc->m_input_index).size() != 3) {
                    continue;
                }
                const auto param = body->get_parameters().at(desc->m_body_parameter_index);
                const auto targets = param->output(0).get_target_inputs();
                const bool feeds_exp = std::any_of(targets.begin(), targets.end(), [](const ov::Input<ov::Node>& t) {
                    return ov::is_type<ov::op::v0::Exp>(t.get_node());
                });
                (feeds_exp ? gate : beta) = param;
            }
            ASSERT_TRUE(gate && beta);
            edit(body, gate, beta);
            body->validate_nodes_and_infer_types();
        }
        model->validate_nodes_and_infer_types();
    }

    // beta is not consumed by the Multiply scaling the delta anymore, but by an unrelated operation.
    static void insert_op_after_beta(const std::shared_ptr<ov::Model>& model) {
        edit_loop_bodies(model, [](const auto&, const auto&, const auto& beta) {
            const auto targets = beta->output(0).get_target_inputs();
            const auto negated = std::make_shared<ov::op::v0::Negative>(beta);
            for (auto target : targets) {
                target.replace_source_output(negated);
            }
        });
    }

    // Inserts the node made by `make` between the updated state and the back edge of every recurrence Loop.
    template <class Make>
    static void insert_op_before_back_edge(const std::shared_ptr<ov::Model>& model, const Make& make) {
        for (const auto& node : model->get_ordered_ops()) {
            const auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node);
            if (!loop) {
                continue;
            }
            const auto body = loop->get_function();
            for (const auto& desc : loop->get_input_descriptions()) {
                const auto merged = ov::as_type_ptr<ov::op::util::SubGraphOp::MergedInputDescription>(desc);
                if (!merged || loop->get_input_partial_shape(desc->m_input_index).size() != 4) {
                    continue;
                }
                const auto result = body->get_results().at(merged->m_body_value_index);
                result->input(0).replace_source_output(make(result->input_value(0)));
            }
            body->validate_nodes_and_infer_types();
        }
        model->validate_nodes_and_infer_types();
    }

    // Number of distinct tensors the masked g / beta Loop inputs are multiplied with.
    static size_t count_ssm_masks(const std::shared_ptr<ov::Model>& model) {
        std::set<ov::Output<ov::Node>> masks;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto loop = ov::as_type_ptr<ov::op::v5::Loop>(node)) {
                for (const auto& desc : loop->get_input_descriptions()) {
                    const auto source = loop->input_value(desc->m_input_index).get_node_shared_ptr();
                    if (ov::is_type<ov::op::v1::Multiply>(source)) {
                        masks.insert(source->input_value(1));
                    }
                }
            }
        }
        return masks.size();
    }

    // std::max(diff, NaN) keeps diff, so a non-finite output must fail the test explicitly.
    static float max_abs_diff(const ov::Tensor& a, const ov::Tensor& b) {
        EXPECT_EQ(a.get_shape(), b.get_shape());
        float diff = 0.f;
        for (size_t i = 0; i < a.get_size(); ++i) {
            const float delta = std::abs(a.data<float>()[i] - b.data<float>()[i]);
            if (!std::isfinite(delta)) {
                ADD_FAILURE() << "non-finite value at element " << i << ": " << a.data<float>()[i] << " vs "
                              << b.data<float>()[i];
                return std::numeric_limits<float>::infinity();
            }
            diff = std::max(diff, delta);
        }
        return diff;
    }

    static bool cpu_available() {
        const auto devices = ov::Core().get_available_devices();
        return std::find(devices.begin(), devices.end(), "CPU") != devices.end();
    }

    static std::shared_ptr<ov::op::v0::Parameter> find_param(const std::shared_ptr<ov::Model>& model,
                                                             const std::string& name) {
        for (const auto& param : model->get_parameters()) {
            if (param->get_output_tensor(0).get_names().count(name)) {
                return param;
            }
        }
        return nullptr;
    }

    // First consumer of the given type of a state parameter, looking through a single Transpose.
    template <class T>
    static ov::Input<ov::Node> state_consumer(const std::shared_ptr<ov::op::v0::Parameter>& param) {
        for (const auto& target : param->output(0).get_target_inputs()) {
            if (ov::is_type<T>(target.get_node())) {
                return target;
            }
            if (ov::is_type<ov::op::v1::Transpose>(target.get_node())) {
                for (const auto& next : target.get_node()->output(0).get_target_inputs()) {
                    if (ov::is_type<T>(next.get_node())) {
                        return next;
                    }
                }
            }
        }
        throw std::runtime_error("no consumer of " + param->get_friendly_name());
    }
};

TEST_F(MaskLinearAttentionPaddingTest, HybridModelIsRewrittenForAnyChunkLength) {
    const auto base = stateless(ov::test::npuw::build_hybrid_llm_test_model());
    const auto n_conv = count_inputs(base, "cache_params.past.conv");
    const auto n_ssm = count_inputs(base, "cache_params.past.ssm");
    ASSERT_GT(n_conv, 0u);
    ASSERT_EQ(n_conv, n_ssm);

    for (const uint32_t chunk : {4u, 8u, 16u}) {
        auto model = make_static(base, chunk);
        const auto scatters_before = count_ops<ov::op::v3::ScatterUpdate>(model);

        ASSERT_TRUE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model)) << "chunk " << chunk;
        ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

        // One ScatterUpdate writing the history per conv state.
        EXPECT_EQ(count_ops<ov::op::v3::ScatterUpdate>(model) - scatters_before, n_conv) << "chunk " << chunk;

        // g and beta of every recurrence Loop are masked.
        EXPECT_EQ(count_masked_loop_inputs(model), 2u * n_ssm) << "chunk " << chunk;
        EXPECT_EQ(count_ssm_masks(model), 1u) << "chunk " << chunk;
    }
}

TEST_F(MaskLinearAttentionPaddingTest, GateConvertedBeforeExpIsRecognized) {
    auto base = stateless(ov::test::npuw::build_hybrid_llm_test_model());
    convert_gate_before_exp(base);
    auto model = make_static(base, 8u);
    const auto n_ssm = count_inputs(model, "cache_params.past.ssm");
    ASSERT_GT(n_ssm, 0u);

    ASSERT_TRUE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(count_masked_loop_inputs(model), 2u * n_ssm);
}

TEST_F(MaskLinearAttentionPaddingTest, ShortConvModelWithoutSsmIsRewritten) {
    const auto base = stateless(ov::test::npuw::build_lfm2_llm_test_model());
    auto model = make_static(base, 8u);
    const auto n_conv = count_inputs(model, "cache_params.past.conv");
    ASSERT_GT(n_conv, 0u);
    ASSERT_EQ(count_inputs(model, "cache_params.past.ssm"), 0u);

    const auto scatters_before = count_ops<ov::op::v3::ScatterUpdate>(model);
    ASSERT_TRUE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(count_ops<ov::op::v3::ScatterUpdate>(model) - scatters_before, n_conv);
}

TEST_F(MaskLinearAttentionPaddingTest, ModelWithoutLinearAttentionIsUntouched) {
    auto model = make_static(stateless(ov::test::npuw::build_llm_test_model()), 8u);
    const auto ops_before = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops_before);
}

// Two Concats consuming the same past conv state and none consuming the other one: the number of
// recognized sites matches the number of states, but a state is not covered.
TEST_F(MaskLinearAttentionPaddingTest, ConvStateWithoutRecognizedSiteLeavesModelUntouched) {
    auto model = make_static(stateless(ov::test::npuw::build_hybrid_llm_test_model()), 8u);
    const auto first = find_param(model, "cache_params.past.conv.0");
    const auto second = find_param(model, "cache_params.past.conv.1");
    ASSERT_TRUE(first && second);
    ASSERT_EQ(first->get_partial_shape(), second->get_partial_shape());

    state_consumer<ov::op::v0::Concat>(second).replace_source_output(first->output(0));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    const auto ops_before = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops_before);
}

TEST_F(MaskLinearAttentionPaddingTest, SsmStateWithoutRecognizedSiteLeavesModelUntouched) {
    auto model = make_static(stateless(ov::test::npuw::build_hybrid_llm_test_model()), 8u);
    const auto first = find_param(model, "cache_params.past.ssm.0");
    const auto second = find_param(model, "cache_params.past.ssm.1");
    ASSERT_TRUE(first && second);
    ASSERT_EQ(first->get_partial_shape(), second->get_partial_shape());

    state_consumer<ov::op::v5::Loop>(second).replace_source_output(first->output(0));
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    const auto ops_before = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops_before);
}

TEST_F(MaskLinearAttentionPaddingTest, BetaNotScalingTheDeltaLeavesModelUntouched) {
    auto model = make_static(stateless(ov::test::npuw::build_hybrid_llm_test_model()), 8u);
    insert_op_after_beta(model);

    const auto ops_before = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops_before);
}

// The decay is applied to a tensor which is not the state carried by the Loop.
TEST_F(MaskLinearAttentionPaddingTest, DecayNotAppliedToCarriedStateLeavesModelUntouched) {
    auto model = make_static(stateless(ov::test::npuw::build_hybrid_llm_test_model()), 8u);
    edit_loop_bodies(model, [](const auto&, const auto& gate, const auto&) {
        const auto exp = (*gate->output(0).get_target_inputs().begin()).get_node();
        const auto unsqueeze = (*exp->output(0).get_target_inputs().begin()).get_node();
        const auto decay = (*unsqueeze->output(0).get_target_inputs().begin()).get_node();
        const size_t state_idx = decay->input_value(0).get_node() == unsqueeze ? 1 : 0;
        const auto& state = decay->input_value(state_idx);
        decay->input(state_idx).replace_source_output(
            ov::op::v0::Constant::create(state.get_element_type(),
                                         state.get_shape(),
                                         std::vector<float>(ov::shape_size(state.get_shape()), 0.f)));
    });

    const auto ops_before = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops_before);
}

// Something else updates the state after the delta is added.
TEST_F(MaskLinearAttentionPaddingTest, UpdateAfterTheDeltaAddLeavesModelUntouched) {
    auto model = make_static(stateless(ov::test::npuw::build_hybrid_llm_test_model()), 8u);
    insert_op_before_back_edge(model, [](const ov::Output<ov::Node>& state) {
        return std::make_shared<ov::op::v0::Negative>(state);
    });

    const auto ops_before = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops_before);
}

TEST_F(MaskLinearAttentionPaddingTest, ConvertOnTheBackEdgeIsRecognized) {
    auto model = make_static(stateless(ov::test::npuw::build_hybrid_llm_test_model()), 8u);
    const auto n_ssm = count_inputs(model, "cache_params.past.ssm");
    insert_op_before_back_edge(model, [](const ov::Output<ov::Node>& state) {
        return std::make_shared<ov::op::v0::Convert>(state, state.get_element_type());
    });

    ASSERT_TRUE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(count_masked_loop_inputs(model), 2u * n_ssm);
}

TEST_F(MaskLinearAttentionPaddingTest, DynamicShapesAreLeftUntouched) {
    auto model = stateless(ov::test::npuw::build_hybrid_llm_test_model());
    const auto ops_before = model->get_ops().size();
    EXPECT_FALSE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_EQ(model->get_ops().size(), ops_before);
}

// The Concat infers a rank-3 output from x alone when the past conv state has a dynamic rank.
TEST_F(MaskLinearAttentionPaddingTest, RankDynamicConvStateLeavesModelUntouched) {
    auto model = make_static(stateless(ov::test::npuw::build_hybrid_llm_test_model()), 8u);
    const auto past = find_param(model, "cache_params.past.conv.0");
    ASSERT_TRUE(past);
    past->set_partial_shape(ov::PartialShape::dynamic());
    ASSERT_NO_THROW(model->validate_nodes_and_infer_types());
    ASSERT_EQ(state_consumer<ov::op::v0::Concat>(past).get_node()->get_output_partial_shape(0).size(), 3u);

    const auto ops_before = model->get_ops().size();
    bool rewritten = true;
    ASSERT_NO_THROW(rewritten = ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    EXPECT_FALSE(rewritten);
    EXPECT_EQ(model->get_ops().size(), ops_before);
}

// A partial chunk padded on the left must leave exactly the same conv/ssm state and the same
// logits as the same tokens processed in a chunk of their own length, for a non-zero carried state.
// The conv state is checked both in its original [B, C, K] layout and re-laid out as [B, K, C], and the
// gate both in the precision of the model and converted from f16 right before Exp.
class MaskLinearAttentionPaddingNumericsTest : public MaskLinearAttentionPaddingTest,
                                               public ::testing::WithParamInterface<std::tuple<uint32_t, bool, bool>> {
};

TEST_P(MaskLinearAttentionPaddingNumericsTest, PaddedChunkMatchesUnpaddedChunk) {
    if (!cpu_available()) {
        GTEST_SKIP() << "CPU plugin is required to execute the models";
    }
    const uint32_t chunk = 8u;
    const auto [padding, relayout_conv, f16_gate] = GetParam();
    ASSERT_LT(padding, chunk);

    auto base = stateless(ov::test::npuw::build_hybrid_llm_test_model());
    if (relayout_conv) {
        ASSERT_TRUE(ov::npuw::util::OptimizeLinCacheLayout().run_on_model(base));
    }
    if (f16_gate) {
        convert_gate_before_exp(base);
    }
    const auto past = make_lincache_past(make_static(base, chunk));

    const auto reference_model = make_static(base, chunk - padding);
    const auto reference = run(reference_model, chunk - padding, 0u, past);

    auto padded_model = make_static(base, chunk);
    const auto padded_without_pass = run(padded_model, chunk, padding, past);
    ASSERT_TRUE(ov::npuw::MaskLinearAttentionPadding().run_on_model(padded_model));
    const auto padded = run(padded_model, chunk, padding, past);

    ASSERT_FALSE(reference.empty());
    float diff_with_pass = 0.f;
    float diff_without_pass = 0.f;
    for (const auto& [name, expected] : reference) {
        ASSERT_TRUE(padded.count(name)) << name;
        if (name == "logits") {
            // Compare the last position only: the earlier ones belong to different chunk layouts.
            const auto vocab = expected.get_shape().back();
            const auto last = [&](const ov::Tensor& t) {
                return ov::Tensor(t,
                                  ov::Coordinate{0, t.get_shape()[1] - 1, 0},
                                  ov::Coordinate{1, t.get_shape()[1], vocab});
            };
            diff_with_pass = std::max(diff_with_pass, max_abs_diff(last(expected), last(padded.at(name))));
            diff_without_pass =
                std::max(diff_without_pass, max_abs_diff(last(expected), last(padded_without_pass.at(name))));
        } else {
            EXPECT_LT(max_abs_diff(expected, padded.at(name)), 1e-4f) << name;
            diff_without_pass = std::max(diff_without_pass, max_abs_diff(expected, padded_without_pass.at(name)));
        }
    }
    EXPECT_LT(diff_with_pass, 1e-4f);
    // The comparison is only meaningful if the unmasked model is actually wrong.
    EXPECT_GT(diff_without_pass, 1e-3f);
}

INSTANTIATE_TEST_SUITE_P(PaddingLength,
                         MaskLinearAttentionPaddingNumericsTest,
                         ::testing::Combine(::testing::Values(1u, 3u, 7u), ::testing::Bool(), ::testing::Bool()));

TEST_F(MaskLinearAttentionPaddingTest, FullChunkIsUnchanged) {
    if (!cpu_available()) {
        GTEST_SKIP() << "CPU plugin is required to execute the models";
    }
    const uint32_t chunk = 8u;
    const auto base = stateless(ov::test::npuw::build_hybrid_llm_test_model());
    const auto past = make_lincache_past(make_static(base, chunk));

    auto model = make_static(base, chunk);
    const auto reference = run(model, chunk, 0u, past);
    ASSERT_TRUE(ov::npuw::MaskLinearAttentionPadding().run_on_model(model));
    const auto masked = run(model, chunk, 0u, past);

    for (const auto& [name, expected] : reference) {
        EXPECT_LT(max_abs_diff(expected, masked.at(name)), 1e-5f) << name;
    }
}

TEST_F(MaskLinearAttentionPaddingTest, ChunkedPrefillPipelineAppliesThePass) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(ov::test::npuw::build_hybrid_llm_test_model(),
                                                     {{"NPUW_LLM_PREFILL_CHUNK_SIZE", "64"}},
                                                     recorder));

    const auto& prefill = require_sub_model(recorder, "_prefill");
    EXPECT_EQ(count_masked_loop_inputs(prefill.model), 2u * count_inputs(prefill.model, "cache_params.past.ssm"));
}

TEST_F(MaskLinearAttentionPaddingTest, WholePrefillDoesNotApplyThePass) {
    RecordingFactory recorder;
    std::unique_ptr<ov::npuw::LLMCompiledModel> compiled;
    ASSERT_NO_THROW(compiled = create_compiled_model(ov::test::npuw::build_hybrid_llm_test_model(),
                                                     {{"NPUW_LLM_PREFILL_CHUNK_SIZE", "0"}},
                                                     recorder));
    const auto& prefill = require_sub_model(recorder, "_prefill");
    EXPECT_EQ(count_masked_loop_inputs(prefill.model), 0u);
}

}  // namespace
