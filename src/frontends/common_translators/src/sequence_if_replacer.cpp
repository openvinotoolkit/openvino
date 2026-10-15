// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sequence_if_replacer.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_at.hpp"
#include "openvino/frontend/sequence_erase.hpp"
#include "openvino/frontend/sequence_insert.hpp"
#include "openvino/frontend/sequence_length.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/identity.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "slot_resolver.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pass {
namespace {

// Strip v16::Identity wrappers to expose the underlying node.
using sal_detail::unwrap_identity;

// Try to extract the concrete sequence elements behind an Output.
// Handles SequenceMark and Identity-of-SequenceMark.
bool try_extract_sequence(const ov::Output<ov::Node>& value, ov::OutputVector& out) {
    const auto unwrapped = unwrap_identity(value);
    auto node = unwrapped.get_node_shared_ptr();
    if (auto seq_mark = ov::as_type_ptr<ov::frontend::SequenceMark>(node)) {
        out = seq_mark->get_sequence();
        return true;
    }
    return false;
}

bool try_resolve_via_sequence_mark(const std::shared_ptr<ov::Node>& helper) {
    ov::OutputVector seq;
    if (!try_extract_sequence(helper->input_value(0), seq))
        return false;
    const auto length = static_cast<int64_t>(seq.size());

    if (auto at = ov::as_type_ptr<ov::frontend::SequenceAt>(helper)) {
        const auto pos_const = ov::util::get_constant_from_source(at->input_value(1));
        if (!pos_const)
            return false;
        const auto pv = pos_const->cast_vector<int64_t>();
        if (pv.size() != 1)
            return false;
        auto idx = pv[0];
        if (idx < 0)
            idx += length;
        if (idx < 0 || idx >= length)
            return false;
        helper->output(0).replace(seq[idx]);
        return true;
    }
    if (ov::is_type<ov::frontend::SequenceLength>(helper)) {
        auto c = v0::Constant::create(ov::element::i64, ov::Shape{}, {length});
        c->set_friendly_name(helper->get_friendly_name());
        ov::copy_runtime_info(helper, c);
        helper->output(0).replace(c->output(0));
        return true;
    }
    if (auto er = ov::as_type_ptr<ov::frontend::SequenceErase>(helper)) {
        int64_t idx = length - 1;
        if (er->has_position()) {
            const auto pos_const = ov::util::get_constant_from_source(er->get_position());
            if (!pos_const)
                return false;
            const auto pv_er = pos_const->cast_vector<int64_t>();
            if (pv_er.size() != 1)
                return false;
            idx = pv_er[0];
            if (idx < 0)
                idx += length;
        }
        if (idx < 0 || idx >= length)
            return false;
        seq.erase(seq.begin() + idx);
        auto new_mark = std::make_shared<ov::frontend::SequenceMark>(seq);
        new_mark->set_friendly_name(er->get_friendly_name());
        ov::copy_runtime_info(er, new_mark);
        helper->output(0).replace(new_mark->output(0));
        return true;
    }
    return false;
}

// Push a SequenceAt / SequenceLength helper into both branches of an If whose
// outputs feed from a SequenceMark in both branches.
bool try_resolve_via_if(const std::shared_ptr<ov::Node>& helper) {
    const auto seq_input = helper->input_value(0);
    const auto if_node = ov::as_type_ptr<v8::If>(seq_input.get_node_shared_ptr());
    if (!if_node)
        return false;

    const auto out_index = seq_input.get_index();
    const auto then_body = if_node->get_then_body();
    const auto else_body = if_node->get_else_body();

    // Locate the Result in each branch that feeds this If output.
    std::shared_ptr<v0::Result> then_result, else_result;
    for (const auto& desc : if_node->get_output_descriptions(v8::If::THEN_BODY_INDEX)) {
        if (desc->m_output_index == out_index) {
            then_result = then_body->get_results().at(desc->m_body_value_index);
            break;
        }
    }
    for (const auto& desc : if_node->get_output_descriptions(v8::If::ELSE_BODY_INDEX)) {
        if (desc->m_output_index == out_index) {
            else_result = else_body->get_results().at(desc->m_body_value_index);
            break;
        }
    }
    if (!then_result || !else_result)
        return false;

    ov::OutputVector then_seq, else_seq;
    if (!try_extract_sequence(then_result->input_value(0), then_seq))
        return false;
    if (!try_extract_sequence(else_result->input_value(0), else_seq))
        return false;
    if (then_seq.size() != else_seq.size())
        return false;
    const auto length = static_cast<int64_t>(then_seq.size());

    if (auto at = ov::as_type_ptr<ov::frontend::SequenceAt>(helper)) {
        const auto pos_const = ov::util::get_constant_from_source(at->input_value(1));
        if (!pos_const)
            return false;
        const auto pv_at = pos_const->cast_vector<int64_t>();
        if (pv_at.size() != 1)
            return false;
        auto idx = pv_at[0];
        if (idx < 0)
            idx += length;
        if (idx < 0 || idx >= length)
            return false;

        // Add new tensor-typed outputs to the If: element[idx] from each branch.
        auto then_new_result = std::make_shared<v0::Result>(then_seq[idx]);
        auto else_new_result = std::make_shared<v0::Result>(else_seq[idx]);
        then_body->add_results({then_new_result});
        else_body->add_results({else_new_result});
        auto new_out = if_node->set_output(then_new_result, else_new_result);
        if_node->validate_and_infer_types();
        ov::copy_runtime_info(helper, {then_new_result, else_new_result});
        helper->output(0).replace(new_out);
        return true;
    }
    if (ov::is_type<ov::frontend::SequenceLength>(helper)) {
        auto c = v0::Constant::create(ov::element::i64, ov::Shape{}, {length});
        c->set_friendly_name(helper->get_friendly_name());
        ov::copy_runtime_info(helper, c);
        helper->output(0).replace(c->output(0));
        return true;
    }
    return false;
}

// After helpers are resolved, dangling sequence-typed If outputs may still
// exist (we add new outputs rather than remove old ones). For an If output
// that has no external consumers and whose body Results' inputs come from a
// frontend sequence helper (SequenceMark / SequenceInsert), redirect those
// Result inputs to a dummy scalar Constant so the helpers become DCE-able.
bool cleanup_dead_if_sequence_output(const std::shared_ptr<v8::If>& if_node) {
    bool changed = false;
    for (size_t out_idx = 0; out_idx < if_node->get_output_size(); ++out_idx) {
        if (!if_node->output(out_idx).get_target_inputs().empty())
            continue;
        std::shared_ptr<v0::Result> then_result, else_result;
        for (const auto& desc : if_node->get_output_descriptions(v8::If::THEN_BODY_INDEX)) {
            if (desc->m_output_index == out_idx) {
                then_result = if_node->get_then_body()->get_results().at(desc->m_body_value_index);
                break;
            }
        }
        for (const auto& desc : if_node->get_output_descriptions(v8::If::ELSE_BODY_INDEX)) {
            if (desc->m_output_index == out_idx) {
                else_result = if_node->get_else_body()->get_results().at(desc->m_body_value_index);
                break;
            }
        }
        auto is_seq_helper = [](const ov::Output<ov::Node>& v) {
            auto n = unwrap_identity(v).get_node_shared_ptr();
            return ov::is_type<ov::frontend::SequenceMark>(n) || ov::is_type<ov::frontend::SequenceInsert>(n) ||
                   ov::is_type<ov::frontend::SequenceErase>(n);
        };
        if (then_result && is_seq_helper(then_result->input_value(0))) {
            auto dummy = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            then_result->input(0).replace_source_output(dummy->output(0));
            changed = true;
        }
        if (else_result && is_seq_helper(else_result->input_value(0))) {
            auto dummy = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
            else_result->input(0).replace_source_output(dummy->output(0));
            changed = true;
        }
    }
    return changed;
}

}  // namespace

// The MatcherPass classes below use OPENVINO_MATCHER_PASS_RTTI, which marks the type with a
// visibility attribute; that requires external linkage, so they must live in a named namespace
// (an anonymous-namespace type has internal linkage and GCC errors with -Werror=attributes).

// MatcherPass (pattern 1): resolve a reader consuming a statically-known
// sequence. The sequence input may be an Identity-wrapped Mark/Insert chain.
//
//   SequenceMark[a,b,c] -> SequenceAt(1)     ==>  b
//   SequenceMark[a,b,c] -> SequenceLength    ==>  Const(3)
//   SequenceMark[a,b,c] -> SequenceErase(1)  ==>  SequenceMark[a,c]
class ResolveSequenceMarkHelper : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pass::ResolveSequenceMarkHelper");
    ResolveSequenceMarkHelper() {
        auto helper = ov::pass::pattern::
            wrap_type<ov::frontend::SequenceAt, ov::frontend::SequenceLength, ov::frontend::SequenceErase>();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            return try_resolve_via_sequence_mark(m.get_match_root());
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(helper, get_type_info().name), callback);
    }
};

// MatcherPass (pattern 2): sink a reader into both branches of an If whose
// corresponding Result yields a statically-known sequence, so the read runs
// per-branch on a plain tensor and the If returns the selected element:
//
//        then: SequenceMark[a0,b0]                  then: a0
//   If <{                          } -> At(0)  ==>  If <{        } -> elem
//        else: SequenceMark[a1,b1]                  else: a1
class PushSequenceHelperIntoIf : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pass::PushSequenceHelperIntoIf");
    PushSequenceHelperIntoIf() {
        auto helper = ov::pass::pattern::wrap_type<ov::frontend::SequenceAt, ov::frontend::SequenceLength>();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            return try_resolve_via_if(m.get_match_root());
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(helper, get_type_info().name), callback);
    }
};

// MatcherPass: after readers resolve, sequence-typed If outputs can be left
// without consumers. Redirect their body Results to a dummy Constant so the
// orphaned SequenceMark/Insert/Erase chains become dead-code-eliminable.
//
//   If output (no consumers): Result <- SequenceMark  ==>  Result <- Const(0)
class CleanupDeadIfSequenceOutputs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pass::CleanupDeadIfSequenceOutputs");
    CleanupDeadIfSequenceOutputs() {
        auto if_pattern = ov::pass::pattern::wrap_type<v8::If>();
        ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
            auto if_node = ov::as_type_ptr<v8::If>(m.get_match_root());
            return if_node && cleanup_dead_if_sequence_output(if_node);
        };
        register_matcher(std::make_shared<ov::pass::pattern::Matcher>(if_pattern, get_type_info().name), callback);
    }
};

bool SequenceIfReplacer::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // The two phases must alternate to a fixpoint: resolving a helper can expose a
    // newly dead If output for cleanup, and cleanup can detach a helper that then
    // lets another resolve. GraphRewrite recurses into nested Loop/If bodies and
    // requeues new nodes, so each phase fully converges over the whole model.
    bool overall_changed = false;
    bool changed = true;
    while (changed) {
        ov::pass::GraphRewrite resolve;
        resolve.add_matcher<ResolveSequenceMarkHelper>();
        resolve.add_matcher<PushSequenceHelperIntoIf>();
        const bool resolved = resolve.run_on_model(model);

        ov::pass::GraphRewrite cleanup;
        cleanup.add_matcher<CleanupDeadIfSequenceOutputs>();
        const bool cleaned = cleanup.run_on_model(model);

        changed = resolved || cleaned;
        overall_changed |= changed;
    }
    return overall_changed;
}

}  // namespace pass
}  // namespace frontend
}  // namespace ov
