// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "duplicate_shared_kv_concat.hpp"

#include <algorithm>
#include <vector>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace npuw {
namespace pass {

namespace {

// Returns true when the parameter name is a past-KV param (contiguous or
// block-split), using the canonical NPUW naming utilities.
bool is_past_kv_param(const std::shared_ptr<ov::op::v0::Parameter>& param) {
    const auto& name = param->get_friendly_name();
    return ov::npuw::util::isPastKeyParam(name) || ov::npuw::util::isPastValueParam(name);
}

// Returns true when all inputs of `concat` except the last are past-KV
// Parameters (or Convert wrapping one).  The final input is the current-chunk
// KV projection and is intentionally left unchecked.
bool has_param_past_inputs(const std::shared_ptr<ov::op::v0::Concat>& concat) {
    const size_t n = concat->get_input_size();
    if (n < 2)
        return false;

    std::vector<std::shared_ptr<ov::op::v0::Parameter>> params;
    params.reserve(n - 1);
    for (size_t i = 0; i + 1 < n; ++i) {
        auto inp = concat->get_input_node_shared_ptr(i);
        if (auto param = ov::as_type_ptr<ov::op::v0::Parameter>(inp)) {
            params.push_back(param);
            continue;
        }
        if (auto cvt = ov::as_type_ptr<ov::op::v0::Convert>(inp)) {
            if (auto param = ov::as_type_ptr<ov::op::v0::Parameter>(cvt->get_input_node_shared_ptr(0))) {
                params.push_back(param);
                continue;
            }
        }
        return false;
    }

    return std::all_of(params.begin(), params.end(), is_past_kv_param);
}

struct KVBroadcastChain {
    std::vector<std::shared_ptr<ov::op::v0::Convert>> converts;
    std::shared_ptr<ov::op::v0::Concat> concat;
    std::shared_ptr<ov::op::v0::Unsqueeze> unsqueeze;  // may be nullptr
    std::shared_ptr<ov::Node> broadcast;               // v1 or v3 Broadcast; may be nullptr
    std::shared_ptr<ov::op::v1::Reshape> reshape;      // fan-out root
};

// Clone the Concat→[Unsqueeze]→[Broadcast]→Reshape chain for every consumer
// beyond the first.  The first consumer keeps the original chain.
void duplicate_for_extra_consumers(const KVBroadcastChain& chain) {
    std::vector<ov::Input<ov::Node>> consumers;
    for (auto& ti : chain.reshape->output(0).get_target_inputs())
        consumers.push_back(ti);

    for (size_t idx = 1; idx < consumers.size(); ++idx) {
        ov::OutputVector new_concat_inputs;
        new_concat_inputs.reserve(chain.concat->get_input_size());
        for (size_t ci = 0; ci < chain.concat->get_input_size(); ++ci) {
            if (chain.converts[ci]) {
                auto new_cvt = chain.converts[ci]->clone_with_new_inputs(chain.converts[ci]->input_values());
                copy_runtime_info(chain.converts[ci], new_cvt);
                new_concat_inputs.push_back(new_cvt->output(0));
            } else {
                new_concat_inputs.push_back(chain.concat->input_value(ci));
            }
        }
        auto new_concat = chain.concat->clone_with_new_inputs(new_concat_inputs);
        copy_runtime_info(chain.concat, new_concat);
        ov::Output<ov::Node> data = new_concat->output(0);

        if (chain.unsqueeze) {
            ov::OutputVector inputs{data};
            for (size_t i = 1; i < chain.unsqueeze->get_input_size(); ++i)
                inputs.push_back(chain.unsqueeze->input_value(i));
            auto new_unsqueeze = chain.unsqueeze->clone_with_new_inputs(inputs);
            copy_runtime_info(chain.unsqueeze, new_unsqueeze);
            data = new_unsqueeze->output(0);
        }

        if (chain.broadcast) {
            ov::OutputVector inputs{data};
            for (size_t i = 1; i < chain.broadcast->get_input_size(); ++i)
                inputs.push_back(chain.broadcast->input_value(i));
            auto new_broadcast = chain.broadcast->clone_with_new_inputs(inputs);
            copy_runtime_info(chain.broadcast, new_broadcast);
            data = new_broadcast->output(0);
        }

        {
            ov::OutputVector inputs{data};
            for (size_t i = 1; i < chain.reshape->get_input_size(); ++i)
                inputs.push_back(chain.reshape->input_value(i));
            auto new_reshape = chain.reshape->clone_with_new_inputs(inputs);
            copy_runtime_info(chain.reshape, new_reshape);
            data = new_reshape->output(0);
        }

        consumers[idx].replace_source_output(data);
    }
}

}  // namespace

DuplicateSharedKVConcat::DuplicateSharedKVConcat() {
    namespace opp = ov::pass::pattern;

    // Pattern: Concat(past-KV params…, current_KV)
    //              → [optional Unsqueeze]
    //              → [optional Broadcast (v1 or v3)]
    //              → Reshape  ← anchored here; predicate checks fan-out and consumer types
    auto p_concat = opp::wrap_type<ov::op::v0::Concat>();
    auto p_unsqueeze = opp::optional<ov::op::v0::Unsqueeze>({p_concat, opp::any_input()});
    auto p_broadcast = opp::optional<ov::op::v1::Broadcast, ov::op::v3::Broadcast>({p_unsqueeze, opp::any_input()});
    auto p_reshape =
        opp::wrap_type<ov::op::v1::Reshape>({p_broadcast, opp::any_input()}, [](const ov::Output<ov::Node>& output) {
            const auto& targets = output.get_target_inputs();
            if (targets.size() <= 1)
                return false;
            for (const auto& ti : targets) {
                auto* n = ti.get_node();
                if (!ov::is_type<ov::op::v0::MatMul>(n) && !ov::is_type<ov::op::v13::ScaledDotProductAttention>(n))
                    return false;
            }
            return true;
        });

    // Note: Use [=] to keep pattern nodes alive in the callback.
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& vm = m.get_pattern_value_map();

        auto matched_concat = ov::as_type_ptr<ov::op::v0::Concat>(vm.at(p_concat).get_node_shared_ptr());
        if (!matched_concat || !has_param_past_inputs(matched_concat))
            return false;

        KVBroadcastChain chain;
        chain.reshape = ov::as_type_ptr<ov::op::v1::Reshape>(vm.at(p_reshape).get_node_shared_ptr());
        chain.concat = matched_concat;

        // opp::optional nodes are absent from vm when not matched (pass-through case).
        // Walk backward from Reshape to detect them instead of using vm.at().
        {
            std::shared_ptr<ov::Node> cur = chain.reshape->get_input_node_shared_ptr(0);
            if (ov::is_type<ov::op::v1::Broadcast>(cur) || ov::is_type<ov::op::v3::Broadcast>(cur)) {
                chain.broadcast = cur;
                cur = cur->get_input_node_shared_ptr(0);
            }
            chain.unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(cur);
        }

        chain.converts.resize(matched_concat->get_input_size());
        // Only past-KV inputs (all but the last) may have a Convert wrapper that needs
        // to be cloned per consumer.  The final input is the current-KV projection and
        // is always shared — leave converts[n-1] as nullptr.
        for (size_t i = 0; i + 1 < matched_concat->get_input_size(); ++i)
            chain.converts[i] = ov::as_type_ptr<ov::op::v0::Convert>(matched_concat->get_input_node_shared_ptr(i));

        const size_t fan_out = chain.reshape->output(0).get_target_inputs().size();
        LOG_DEBUG("DuplicateSharedKVConcat: Reshape \""
                  << chain.reshape->get_friendly_name() << "\" fan-out=" << fan_out
                  << " — duplicating KV broadcast chain for " << (fan_out - 1) << " extra consumer(s)");

        duplicate_for_extra_consumers(chain);
        return true;
    };

    register_matcher(std::make_shared<opp::Matcher>(p_reshape, "DuplicateSharedKVConcat"), std::move(callback));
}

}  // namespace pass
}  // namespace npuw
}  // namespace ov
