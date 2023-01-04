// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_exp_reduceop_to_keep_in_mixed_precision.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/op/or.hpp>

#include "itt.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "transformations/common_optimizations/mark_subgraphs_to_keep_in_mixed_precision.hpp"
#include "transformations/rt_info/reduceop_path.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;

namespace ov {
namespace pass {

class InitMarkReduceOpExp : public pass::MatcherPass {
public:
    OPENVINO_RTTI("InitMarkReduceOpExp", "0");
    InitMarkReduceOpExp() {
        MATCHER_SCOPE(InitMarkReduceOpExp);

        auto reduce_ops = pattern::wrap_type<opset3::ReduceSum, opset3::ReduceMean>();

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;
            mark_reduceop_path(node);
            return true;
        };
        auto m = make_shared<pattern::Matcher>(reduce_ops, matcher_name);
        register_matcher(m, callback);
    }
};

class PropagateUpMarkReduceOpExp : public pass::MatcherPass {
public:
    OPENVINO_RTTI("PropagateUpMarkReduceOpExp", "0");
    PropagateUpMarkReduceOpExp() {
        MATCHER_SCOPE(PropagateUpMarkReduceOpExp);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            bool has_marked_output = false;
            for (const auto& output : node->outputs()) {
                for (const auto& out_inputs : output.get_target_inputs()) {
                    if (out_inputs.get_element_type().is_real() &&
                        is_reduceop_path(out_inputs.get_node()->shared_from_this())) {
                        has_marked_output = true;
                    }
                }
            }
            if (!has_marked_output)
                return false;

            mark_reduceop_path(node);
            return true;
        };
        auto m = make_shared<pattern::Matcher>(propagate_through_ops, matcher_name);
        register_matcher(m, callback);
    }
};

MarkExpReduceOpToKeepInMixedPrecision::MarkExpReduceOpToKeepInMixedPrecision() {
    ADD_MATCHER_FOR_THIS(InitMarkReduceOpExp)
    ADD_MATCHER_FOR_THIS(PropagateUpMarkReduceOpExp)
}
}  // namespace pass
}  // namespace ov
