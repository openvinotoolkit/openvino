// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_select_mask_fusion.hpp"

#include <limits>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
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
// Minimum magnitude for the Select "else" constant to be treated as -inf.
// Attention masks frequently saturate -inf to -FP16_MAX (-65504) during
// precision conversion; any value at or below this threshold softmaxes to ~0,
// so rewriting it into an additive mask stays equivalent.
constexpr float kNegInfThreshold = -1e4f;

bool is_softmax(const ov::Node* n) {
    return ov::is_type<v1::Softmax>(n) || ov::is_type<v8::Softmax>(n);
}

bool feeds_softmax(const ov::Output<ov::Node>& out) {
    for (const auto& in : out.get_target_inputs()) {
        auto* consumer = in.get_node();
        if (is_softmax(consumer))
            return true;
        if (ov::is_type<v1::Reshape>(consumer)) {
            for (const auto& in2 : consumer->output(0).get_target_inputs()) {
                if (is_softmax(in2.get_node()))
                    return true;
            }
        }
    }
    return false;
}
}  // namespace

ov::intel_gpu::SDPASelectMaskFusion::SDPASelectMaskFusion() {
    // Decomposed attention with a where-style mask:
    //   ... -> Multiply(scale) -> Select(mask, scores, neg_inf) -> Softmax -> ...
    // Convert it to the additive form the common ov::pass::SDPAFusion understands:
    //   Select(mask, scores, neg_inf) == scores + Select(mask, 0, neg_inf)
    // valid because the following Softmax normalizes the masked-out (-inf) entries to 0.
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

        // scores must have a real element type to build a matching additive mask.
        const auto et = scores_out.get_element_type();
        if (!et.is_real())
            return false;

        // The masked-out (else) value must be a scalar constant acting as -inf.
        auto neg_inf_const = ov::as_type_ptr<v0::Constant>(pm.at(neg_inf).get_node_shared_ptr());
        if (!neg_inf_const || ov::shape_size(neg_inf_const->get_shape()) != 1)
            return false;
        const float neg_inf_val = neg_inf_const->cast_vector<float>()[0];
        // Accept values down to the threshold to accommodate -inf saturated to -FP16_MAX (-65504).
        if (neg_inf_val > kNegInfThreshold)
            return false;

        // The rewrite is only equivalent once normalized by a following Softmax.
        if (!feeds_softmax(pm.at(select_m)))
            return false;

        // additive_mask = Select(mask, 0, -inf); result = scores + additive_mask.
        auto zero = v0::Constant::create(et, ov::Shape{}, {0.0f});
        auto neg_inf_new = v0::Constant::create(et, ov::Shape{}, {-std::numeric_limits<float>::infinity()});
        auto add_mask = std::make_shared<v1::Select>(cond_out, zero, neg_inf_new);
        auto add = std::make_shared<v1::Add>(scores_out, add_mask);
        add->set_friendly_name(select_node->get_friendly_name());
        ov::copy_runtime_info(select_node, {zero, neg_inf_new, add_mask, add});
        ov::replace_node(select_node, add);
        return true;
    };

    auto m = std::make_shared<Matcher>(select_m, "SDPASelectMaskFusion");
    register_matcher(m, callback);
}
