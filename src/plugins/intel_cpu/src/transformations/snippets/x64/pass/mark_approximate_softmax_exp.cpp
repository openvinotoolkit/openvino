// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mark_approximate_softmax_exp.hpp"

#include <cstddef>
#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "utils/rt_info/approximate_exp_attribute.hpp"

ov::intel_cpu::pass::MarkApproximateSoftmaxExp::MarkApproximateSoftmaxExp() {
    MATCHER_SCOPE(MarkApproximateSoftmaxExp);
    auto multiply_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>();

    auto callback = [](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::MarkApproximateSoftmaxExp")
        const auto multiply = m.get_match_root();

        std::shared_ptr<ov::Node> exp;
        std::shared_ptr<ov::snippets::op::PowerStatic> reciprocal;
        for (size_t i = 0; i < multiply->get_input_size(); ++i) {
            const auto operand = multiply->get_input_node_shared_ptr(i);
            if (ov::is_type<ov::op::v0::Exp>(operand)) {
                exp = operand;
            } else if (const auto power = ov::as_type_ptr<ov::snippets::op::PowerStatic>(operand)) {
                reciprocal = power;
            }
        }
        if (exp == nullptr || reciprocal == nullptr || reciprocal->get_power() != -1.F) {
            return false;
        }

        // The reciprocal has to be of a sum of this very exp, otherwise the numerator and the
        // denominator do not share the approximation and the error does not stay on a ratio.
        const auto reduce_sum = ov::as_type_ptr<ov::snippets::op::ReduceSum>(reciprocal->get_input_node_shared_ptr(0));
        if (reduce_sum == nullptr || reduce_sum->get_input_node_shared_ptr(0) != exp) {
            return false;
        }

        // Any consumer beyond the row sum and the normalising multiply would see the approximation
        // on a raw exponential instead.
        if (exp->get_output_target_inputs(0).size() != 2) {
            return false;
        }

        mark_as_approximate_exp(exp);
        // The graph is unchanged: only run-time information was added.
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(multiply_m, matcher_name);
    register_matcher(m, callback);
}
