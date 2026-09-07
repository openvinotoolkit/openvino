// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/make_stateful.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/variable.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::pass {

namespace {

// The cache is no longer part of the model's IO once rewritten: its Parameter is now a ReadValue
// and its Result an Assign sink. Results must go first so the Parameters have no consumers left.
void finalize_stateful_rewrite(const std::shared_ptr<ov::Model>& model,
                               const ov::ResultVector& results_to_remove,
                               const ov::SinkVector& new_sinks,
                               const ov::ParameterVector& params_to_remove) {
    for (const auto& r : results_to_remove) {
        model->remove_result(r);
    }
    model->add_sinks(new_sinks);
    for (const auto& p : params_to_remove) {
        model->remove_parameter(p);
    }
}

// The axis the cache grows along. An un-preallocated cache Parameter states it by construction: it
// is the one dynamic axis (the token count), every other being a static batch / head / head-size.
// A caller that preallocates the cache must say which axis it is, since none is dynamic then.
int64_t resolve_append_axis(const ov::PartialShape& ps, const std::string& cache_name, int64_t requested) {
    const int64_t rank = ps.rank().get_length();
    if (requested >= 0) {
        OPENVINO_ASSERT(requested < rank,
                        "[GGUF] GGUFMakeStateful: append axis ",
                        requested,
                        " is out of range for cache '",
                        cache_name,
                        "' of shape ",
                        ps);
        return requested;
    }
    int64_t axis = -1;
    for (int64_t i = 0; i < rank; ++i) {
        if (ps[i].is_dynamic()) {
            OPENVINO_ASSERT(axis < 0,
                            "[GGUF] GGUFMakeStateful: cache '",
                            cache_name,
                            "' has shape ",
                            ps,
                            " with more than one dynamic axis, so its token axis cannot be inferred; "
                            "construct the pass with an explicit append_axis");
            axis = i;
        }
    }
    OPENVINO_ASSERT(axis >= 0,
                    "[GGUF] GGUFMakeStateful: cache '",
                    cache_name,
                    "' has the fully static shape ",
                    ps,
                    ", so its token axis cannot be inferred; construct the pass with an explicit append_axis");
    return axis;
}

}  // namespace

const std::string& gguf_recurrent_states_key() {
    static const std::string key = "gguf_recurrent_states";
    return key;
}

const std::string& gguf_imrope_key() {
    static const std::string key = "gguf_is_imrope";
    return key;
}

const std::string& gguf_swa_window_key() {
    static const std::string key = "gguf_swa_window_size";
    return key;
}

// Rewrite the recurrent (overwritten, non-appending) states into OpenVINO Variables.
//
// A KV cache is found by walking the SetRows writes and is grown with a Concat along its token
// axis. A recurrent state has neither: it is read whole at the start of a step and replaced whole
// at the end, so the graph carries nothing that marks it and the pairing arrives via rt_info (see
// gguf_recurrent_states_key). The rewrite is correspondingly simpler -- ReadValue -> ... -> Assign
// with no Concat, and no beam Gather, since there is no past to reorder.
//
// Their shapes are fully static, so the init is a real zeros Constant. Zero is also the correct
// initial value: ggml starts a sequence with a zeroed conv window and delta matrix.
static bool make_recurrent_states_stateful(const std::shared_ptr<ov::Model>& model) {
    using namespace ov::op;
    auto it = model->get_rt_info().find(gguf_recurrent_states_key());
    if (it == model->get_rt_info().end()) {
        return false;
    }
    const auto& flat = it->second.as<std::vector<std::string>>();
    if (flat.empty() || flat.size() % 2 != 0) {
        return false;
    }

    ov::ParameterVector params_to_remove;
    ov::ResultVector results_to_remove;
    ov::SinkVector new_sinks;

    for (size_t i = 0; i + 1 < flat.size(); i += 2) {
        const std::string& in_name = flat[i];
        const std::string& out_name = flat[i + 1];

        auto param = find_parameter(model, in_name);
        if (!param) {
            continue;  // already rewritten (a second run of this pass)
        }
        const auto& ps = param->get_partial_shape();
        OPENVINO_ASSERT(ps.is_static(),
                        "[GGUF] GGUFMakeStateful: recurrent state '",
                        in_name,
                        "' must have a fully static shape, got ",
                        ps);

        // The Result holding this state's new value: the one whose producing node carries the
        // state's output name. The builder names that node after the state (see the VIEW cases),
        // which is why those names have to survive translation.
        std::shared_ptr<v0::Result> state_result;
        for (const auto& r : model->get_results()) {
            const auto producer = r->get_input_node_shared_ptr(0);
            if (producer->get_friendly_name().find(out_name) != std::string::npos) {
                state_result = r;
                break;
            }
        }
        OPENVINO_ASSERT(state_result, "[GGUF] GGUFMakeStateful: no Result produces recurrent state '", out_name, "'");

        const auto et = param->get_element_type();
        auto var = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{ps, et, in_name});
        auto init = v0::Constant::create(et, ps.to_shape(), std::vector<float>(1, 0.0f));
        auto read_value = std::make_shared<v6::ReadValue>(init, var);
        read_value->set_friendly_name(in_name);
        ov::replace_node(param, read_value);

        new_sinks.push_back(std::make_shared<v6::Assign>(state_result->input_value(0), var));
        model->add_variables({var});
        results_to_remove.push_back(state_result);
        params_to_remove.push_back(param);
    }

    if (params_to_remove.empty()) {
        return false;
    }
    finalize_stateful_rewrite(model, results_to_remove, new_sinks, params_to_remove);
    return true;
}

bool GGUFMakeStateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    using namespace ov::op;
    // beam_idx reorders the past cache along the batch axis for beam search (identity at batch 1 /
    // beam_idx [0], but emitting it is what lets CPU's stateful_sdpa_fusion match). It belongs to
    // the STATE, so this pass owns it: it indexes an OpenVINO cache that ggml has no equivalent
    // of, so no decoder should declare it. Created here, next to its only consumer (the Gather
    // below); a model that already has one (a caller that declared it, or a second run of this
    // pass) keeps it.
    auto beam_idx = find_parameter(model, m_beam_idx_name);
    const bool created_beam_idx = beam_idx == nullptr;
    if (created_beam_idx) {
        beam_idx = std::make_shared<v0::Parameter>(ov::element::i32, ov::PartialShape{ov::Dimension()});
        beam_idx->set_friendly_name(m_beam_idx_name);
        beam_idx->output(0).set_names({m_beam_idx_name});
    }

    // Only a SetRows writing into a model Parameter is a cache write; the rest (e.g. MoE routing
    // writes) are left to the default stateless lowering that runs after this pass. Collect first,
    // then rewrite, so the graph is not mutated while being walked.
    std::vector<std::shared_ptr<SetRows>> cache_writes;
    for (const auto& node : model->get_ops()) {
        auto set_rows = ov::as_type_ptr<SetRows>(node);
        if (!set_rows) {
            continue;
        }
        auto dst = ov::as_type_ptr<v0::Parameter>(set_rows->input_value(2).get_node_shared_ptr());
        if (dst && !m_skip_caches.count(dst->get_friendly_name())) {
            cache_writes.push_back(set_rows);
        }
    }
    // A model can have recurrent states and no KV cache at all (an all-linear-attention stack),
    // so the recurrent rewrite must not sit behind this early return.
    if (cache_writes.empty()) {
        return make_recurrent_states_stateful(model);
    }

    ov::ParameterVector params_to_remove;
    ov::ResultVector results_to_remove;
    ov::SinkVector new_sinks;
    // Same beam_idx Gather axis for every cache; hoisted out of the loop below.
    auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

    for (const auto& set_rows : cache_writes) {
        auto new_rows = set_rows->input_value(0);
        auto cache_param = ov::as_type_ptr<v0::Parameter>(set_rows->input_value(2).get_node_shared_ptr());
        const auto& cache_name = cache_param->get_friendly_name();
        const auto& ps = cache_param->get_partial_shape();
        const auto et = cache_param->get_element_type();
        OPENVINO_ASSERT(ps.rank().is_static(),
                        "[GGUF] GGUFMakeStateful requires a static cache rank, got ",
                        ps,
                        " for '",
                        cache_name,
                        "'");
        const int64_t axis = resolve_append_axis(ps, cache_name, m_append_axis);

        // The state holds however many tokens have accumulated, so the append axis is dynamic on the
        // Variable and its initial extent is 0 (no past on the first inference). Every other axis
        // keeps the Parameter's declared dimension and so must be static to build the init constant.
        ov::PartialShape var_shape = ps;
        var_shape[axis] = ov::Dimension::dynamic();
        auto var = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, et, cache_name});

        ov::Shape init_shape;
        for (int64_t i = 0; i < ps.rank().get_length(); ++i) {
            if (i == axis) {
                init_shape.push_back(0);
                continue;
            }
            OPENVINO_ASSERT(ps[i].is_static(),
                            "[GGUF] GGUFMakeStateful requires static non-token cache dims, got ",
                            ps,
                            " for '",
                            cache_name,
                            "'");
            init_shape.push_back(static_cast<size_t>(ps[i].get_length()));
        }
        // Empty init: required, not cosmetic -- CPU's MemoryInputSDPA aborts on a MemoryInput with
        // zero parent edges (see the header note).
        auto init = v0::Constant::create(et, init_shape, std::vector<float>{});
        auto read_value = std::make_shared<v6::ReadValue>(init, var);

        // The SetRows placeholder presents the new rows flattened to [.., 1, tokens, row_size] (see
        // translate_set_rows), which need not be the cache's own split of those same elements -- e.g.
        // a [1, tokens, n_head_kv, head_size] cache receives [1, 1, tokens, n_head_kv*head_size]. So
        // re-split them against the cache layout before the Concat: the token axis is -1, the axes
        // after it take the cache's static dims, and the axes before it are copied from the incoming
        // data (special_zero's 0) to stay valid under either token-axis layout.
        std::vector<int64_t> split_pattern;
        for (int64_t i = 0; i < ps.rank().get_length(); ++i) {
            split_pattern.push_back(i < axis ? 0 : (i == axis ? -1 : ps[i].get_length()));
        }
        auto split_shape = v0::Constant::create(ov::element::i64, {split_pattern.size()}, split_pattern);
        new_rows = std::make_shared<v1::Reshape>(new_rows, split_shape, true);

        // Reorder the past by beam_idx before appending, so each beam continues its own history.
        auto past = std::make_shared<v8::Gather>(read_value, beam_idx, axis0);
        auto concat = std::make_shared<v0::Concat>(ov::OutputVector{past, new_rows}, axis);
        concat->set_friendly_name(set_rows->get_friendly_name());
        new_sinks.push_back(std::make_shared<v6::Assign>(concat, var));

        // The stateless graph returns each updated cache as a Result; in the stateful form the Assign
        // sink above takes that role, so those Results go. Identify them as the Results reading THIS
        // write -- not by matching the cache's name, which only happens to work while the builder
        // names a cache's write after the cache itself. Collect them before replace_node, while the
        // SetRows is still the node they read.
        for (const auto& consumer : set_rows->output(0).get_target_inputs()) {
            if (auto r = ov::as_type_ptr<v0::Result>(consumer.get_node()->shared_from_this())) {
                results_to_remove.push_back(r);
            }
        }

        // Every remaining consumer of the write (attention's read of the cache) now reads the grown
        // Concat instead.
        ov::replace_node(set_rows, concat);
        model->add_variables({var});

        params_to_remove.push_back(cache_param);
    }

    // Only now, having actually built the Gathers that read it -- so a pass that converted nothing
    // adds no input.
    if (created_beam_idx) {
        model->add_parameters({beam_idx});
    }
    finalize_stateful_rewrite(model, results_to_remove, new_sinks, params_to_remove);

    // Recurrent states are independent of the KV caches; a hybrid stack (qwen35) has both.
    make_recurrent_states_stateful(model);

    model->validate_nodes_and_infer_types();
    return true;
}

}  // namespace ov::frontend::gguf::pass
