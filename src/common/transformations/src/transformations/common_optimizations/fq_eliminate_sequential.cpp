// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_eliminate_sequential.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/squeeze_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

namespace ov::pass {

namespace {

bool have_same_fake_quantize_params(const std::shared_ptr<v0::FakeQuantize>& lhs,
                                    const std::shared_ptr<v0::FakeQuantize>& rhs) {
    if (!lhs || !rhs || lhs->get_levels() != rhs->get_levels() ||
        lhs->get_auto_broadcast() != rhs->get_auto_broadcast()) {
        return false;
    }

    for (size_t index = 1; index < lhs->get_input_size(); ++index) {
        if (!ov::compare_constants(lhs->input_value(index).get_node_shared_ptr(),
                                   rhs->input_value(index).get_node_shared_ptr())) {
            return false;
        }
    }

    return true;
}

}  // namespace

FakeQuantizeEliminateSequential::FakeQuantizeEliminateSequential() {
    MATCHER_SCOPE(FakeQuantizeEliminateSequential);
    auto p_fq1 = pattern::wrap_type<v0::FakeQuantize>(
        {pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()},
        pattern::consumers_count(1));

    // Eliminates a redundant second FakeQuantize (FQ1 -> FQ2). Notation below:
    // FQ(in_low, in_high, out_low, out_high, levels). FQ1 and FQ2 may be separated by one or several
    // value-preserving Reshape/Transpose/Squeeze/Unsqueeze ops: a scalar (per-tensor) FakeQuantize is
    // applied element-wise, so it commutes with such ops and the folding stays valid through the chain.
    // FQ2 is dropped (FQ1 kept) when it is identical to FQ1 (same range constants and levels), e.g.
    //   FQ1(-1, 1, -1, 1, 256) -> FQ2(-1, 1, -1, 1, 256)  =>  FQ1(-1, 1, -1, 1, 256)
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq1 = ov::as_type_ptr<v0::FakeQuantize>(m.get_match_root());
        if (!fq1) {
            return false;
        }

        // Walk down from FQ1 through value-preserving Reshape/Transpose/Squeeze/Unsqueeze ops to reach
        // the consuming FQ2. Every op along the chain (starting with FQ1) must have a single consumer,
        // otherwise bypassing it would change the graph for its other consumers.
        auto is_value_preserving = [](const std::shared_ptr<ov::Node>& node) {
            return ov::is_type<v1::Reshape>(node) || ov::is_type<v1::Transpose>(node) ||
                   ov::is_type<op_util::SqueezeBase>(node) || ov::is_type<v0::Unsqueeze>(node);
        };
        auto output = fq1->output(0);
        std::shared_ptr<ov::Node> consumer;
        while (true) {
            if (output.get_target_inputs().size() != 1) {
                return false;
            }
            consumer = output.get_target_inputs().begin()->get_node()->shared_from_this();
            if (!is_value_preserving(consumer)) {
                break;
            }
            output = consumer->output(0);
        }
        auto fq2 = ov::as_type_ptr<v0::FakeQuantize>(consumer);
        if (!fq2) {
            return false;
        }

        // Drop FQ2 only when it is identical to FQ1: same levels and same four range bounds. Then FQ2
        // re-applies the exact quantization already produced by FQ1 and is redundant.
        if (!have_same_fake_quantize_params(fq1, fq2)) {
            return false;
        }
        return replace_output_update_name(fq2->output(0), fq2->input_value(0));
    };

    auto m = std::make_shared<pattern::Matcher>(p_fq1, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::pass
