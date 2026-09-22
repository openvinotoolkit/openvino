// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_select_mask_fusion.hpp"

#include <limits>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::consumers_count;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

namespace {
// Sentinel upper bound: accept masked-out values at or below the lowest f16 value (-65504). In
// f16 this is exactly the value a framework masked_fill(-inf) saturates to (or a true -inf); in
// bf16 it also covers the bf16 saturation (about -3.39e38) plus large finite sentinels that are
// still numerically safe once mask_value is added. A weaker sentinel (e.g. -1e4) is not
// guaranteed to stay below the surviving scores, so it keeps its Select.
constexpr float kMaskSentinelMax = -65504.0f;

// Hard requirements of the attention Softmax of the common ov::pass::SDPAFusion, which this
// rewrite relies on for the inserted Add to be fused: v8 Softmax (the v1 -> v8 upgrade is disabled
// in CommonOptimizations, so a v1 Softmax would never be fused), the axis on the last dimension,
// and a single consumer.
bool is_attention_softmax(const ov::Node* n) {
    if (!ov::is_type<v8::Softmax>(n))
        return false;
    const auto* softmax = static_cast<const v8::Softmax*>(n);
    if (softmax->get_output_target_inputs(0).size() != 1)
        return false;
    const auto rank = softmax->get_input_partial_shape(0).rank();
    if (!rank.is_static())
        return false;
    const auto axis = ov::util::try_normalize_axis(softmax->get_axis(), rank, *softmax);
    return axis == static_cast<size_t>(rank.get_length() - 1);
}

// Additive mask value: half of the lowest representable score value. A true -inf cannot be built
// for f16/bf16 constants and `scores + value` cannot overflow to -inf, so a fully masked row stays
// finite instead of turning into NaN inside the Softmax.
double additive_mask_value(const ov::element::Type& et) {
    if (et == ov::element::f16)
        return static_cast<double>(std::numeric_limits<ov::float16>::lowest()) / 2.0;
    if (et == ov::element::bf16)
        return static_cast<double>(std::numeric_limits<ov::bfloat16>::lowest()) / 2.0;
    OPENVINO_THROW("Unsupported score element type for the SDPA additive mask: ", et.get_type_name());
}

constexpr size_t kMaxMatMulLookback = 4;

bool produced_by_matmul(const ov::Output<ov::Node>& out, size_t depth = 0) {
    const auto node = out.get_node_shared_ptr();
    if (const auto matmul = ov::as_type_ptr<v0::MatMul>(node)) {
        return !matmul->get_transpose_a() && matmul->get_output_target_inputs(0).size() == 1;
    }

    if (depth >= kMaxMatMulLookback || (!ov::is_type<v1::Multiply>(node) && !ov::is_type<v1::Divide>(node) &&
                                        !ov::is_type<v1::Reshape>(node) && !ov::is_type<v0::Convert>(node)))
        return false;

    // Either operand of the scaling op can hold the scores, so both inputs are followed.
    for (const auto& input : node->inputs()) {
        const auto src = input.get_source_output();
        if (!ov::is_type<v0::Constant>(src.get_node_shared_ptr()) && produced_by_matmul(src, depth + 1))
            return true;
    }
    return false;
}

template <typename Pred>
bool any_consumer(const ov::Node* node, const Pred& pred) {
    for (const auto& in : node->output(0).get_target_inputs()) {
        const auto* consumer = in.get_node();
        if (pred(consumer))
            return true;

        if (ov::is_type<v1::Reshape>(consumer)) {
            for (const auto& in2 : consumer->output(0).get_target_inputs()) {
                if (pred(in2.get_node()))
                    return true;
            }
        }
    }
    return false;
}

bool softmax_feeds_matmul(const ov::Node* softmax) {
    return any_consumer(softmax, [](const ov::Node* n) {
        if (!ov::is_type<v0::MatMul>(n))
            return false;

        const auto* matmul = static_cast<const v0::MatMul*>(n);
        // The final MatMul(probs, V) of the common ov::pass::SDPAFusion pattern uses plain,
        // non-transposed operands.
        return !matmul->get_transpose_a() && !matmul->get_transpose_b();
    });
}

bool feeds_attention_softmax(const ov::Output<ov::Node>& out) {
    return any_consumer(out.get_node(), [](const ov::Node* n) { return is_attention_softmax(n) && softmax_feeds_matmul(n); });
}
}  // namespace

ov::intel_gpu::SDPASelectMaskFusion::SDPASelectMaskFusion() {
    // Decomposed attention with a where-style mask:
    //   MatMul(Q,K) -> [scale] -> Select(mask, scores, sentinel) -> Softmax -> MatMul(probs,V)
    // is rewritten to the additive form the common ov::pass::SDPAFusion understands:
    //   scores + Select(mask, 0, mask_value)
    //
    // The rewrite is not an exact equivalence of the Select itself (a masked entry becomes
    // `scores + mask_value` instead of the sentinel); it is only safe once normalized by the
    // following Softmax. With mask_value = half of the lowest value of the score precision, a
    // masked entry underflows to 0 in the Softmax for realistic attention score ranges.
    // Remaining differences from the original Select, which a constant additive value cannot
    // close: a masked score far above every surviving score (near the type maximum) is not
    // guaranteed to underflow; a NaN masked score stays NaN (the Select would have returned
    // the sentinel); and a fully masked row becomes score-weighted instead of uniform when the
    // sentinel is finite.
    //
    // The whole attention pattern with the hard requirements of the common ov::pass::SDPAFusion
    // (v8 Softmax on the last axis with a single consumer, plain MatMuls, scores rank 2..4) is
    // required so the inserted Add is guaranteed to be fused; unrelated Select -> Softmax graphs
    // keep their Select.
    auto cond = any_input();
    auto scores = any_input();
    auto neg_inf = wrap_type<v0::Constant>();
    auto select_m = wrap_type<v1::Select>({cond, scores, neg_inf}, consumers_count(1));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto select_node = ov::as_type_ptr<v1::Select>(pm.at(select_m).get_node_shared_ptr());
        if (!select_node)
            return false;

        const auto cond_out = pm.at(cond);
        const auto scores_out = pm.at(scores);

        if (cond_out.get_element_type() != ov::element::boolean)
            return false;

        if (select_node->get_auto_broadcast() != ov::op::AutoBroadcastType::NUMPY)
            return false;

        const auto et = scores_out.get_element_type();
        if (et != ov::element::f16 && et != ov::element::bf16)
            return false;

        // The common ov::pass::SDPAFusion matches scores of rank 2..4 (the mask forces the
        // rank <= 4 bound) and a mask of rank <= 4.
        const auto scores_rank = scores_out.get_partial_shape().rank();
        const auto cond_rank = cond_out.get_partial_shape().rank();
        if (!scores_rank.is_static() || scores_rank.get_length() < 2 || scores_rank.get_length() > 4 ||
            !cond_rank.is_static() || cond_rank.get_length() > 4)
            return false;

        const double mask_value = additive_mask_value(et);

        auto neg_inf_const = ov::as_type_ptr<v0::Constant>(pm.at(neg_inf).get_node_shared_ptr());
        if (!neg_inf_const || ov::shape_size(neg_inf_const->get_shape()) != 1)
            return false;
        // A true -inf passes this comparison as well; NaN does not, so it is rejected.
        if (!(neg_inf_const->cast_vector<float>()[0] <= kMaskSentinelMax))
            return false;

        if (!produced_by_matmul(scores_out) || !feeds_attention_softmax(pm.at(select_m)))
            return false;

        auto zero = v0::Constant::create(et, ov::Shape{}, {0.0});
        auto neg_inf_new = v0::Constant::create(et, ov::Shape{}, {mask_value});
        auto add_mask = std::make_shared<v1::Select>(cond_out, zero, neg_inf_new, ov::op::AutoBroadcastType::NUMPY);
        auto add = std::make_shared<v1::Add>(scores_out, add_mask, ov::op::AutoBroadcastType::NUMPY);
        add->set_friendly_name(select_node->get_friendly_name());
        ov::copy_runtime_info(select_node, {zero, neg_inf_new, add_mask, add});
        ov::replace_node(select_node, add);
        return true;
    };

    auto m = std::make_shared<Matcher>(select_m, "SDPASelectMaskFusion");
    register_matcher(m, callback);
}
