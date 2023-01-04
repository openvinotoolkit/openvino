// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/util/broadcast_base.hpp>
#include <ngraph/op/util/gather_base.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/op/or.hpp>

#include "itt.hpp"
#include "ngraph/env_util.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/mark_div_with_eps_to_keep_in_mixed_precision.hpp"
#include "transformations/common_optimizations/mark_subgraphs_to_keep_in_mixed_precision.hpp"
#include "transformations/common_optimizations/mark_exp_reduceop_to_keep_in_mixed_precision.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/reduceop_path.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;

namespace ov {
namespace pass {

class InitMarkToKeepInMixedPrecision : public pass::MatcherPass {
public:
    OPENVINO_RTTI("InitMarkToKeepInMixedPrecision", "0");
    InitMarkToKeepInMixedPrecision() {
        MATCHER_SCOPE(InitMarkToKeepInMixedPrecision);
        auto ops_to_be_kept_fp32 = pattern::wrap_type<opset3::MVN, opset8::MVN, opset8::NormalizeL2, opset8::Exp>();

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            if (auto exp_op = dynamic_pointer_cast<opset3::Exp>(node) && !is_reduceop_path(node))
                return false;

            disable_fp16_compression(node);
            return true;
        };
        auto m = make_shared<pattern::Matcher>(ops_to_be_kept_fp32, matcher_name);
        register_matcher(m, callback);
    }
};

std::shared_ptr<Node> propagate_through_ops = pattern::wrap_type<opset8::Squeeze,
                                            opset8::Unsqueeze,
                                            opset8::Reshape,
                                            op::util::BroadcastBase,
                                            op::util::BinaryElementwiseArithmetic,
                                            op::util::UnaryElementwiseArithmetic,
                                            opset8::MVN,
                                            opset3::MVN,
                                            opset8::NormalizeL2,
                                            opset8::Sqrt,
                                            opset8::StridedSlice,
                                            opset8::ReduceSum,
                                            opset8::ReduceMean,
                                            opset8::Slice,
                                            opset8::VariadicSplit,
                                            opset8::Split,
                                            op::util::GatherBase,
                                            opset8::Concat,
                                            opset8::Convert, // through Convert can go only to Constants
                                            opset8::Constant,
                                            opset8::Tile>();

class PropagateUpMarkToKeepInMixedPrecision : public pass::MatcherPass {
public:
    OPENVINO_RTTI("PropagateUpMarkToKeepInMixedPrecision", "0");
    PropagateUpMarkToKeepInMixedPrecision() {
        MATCHER_SCOPE(PropagateUpMarkToKeepInMixedPrecision);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            bool has_marked_output = false;
            for (const auto& output : node->outputs()) {
                for (const auto &out_inputs: output.get_target_inputs()) {
                    if (out_inputs.get_element_type().is_real() &&
                            fp16_compression_is_disabled(out_inputs.get_node()->shared_from_this())) {
                        has_marked_output = true;
                    }
                }
            }

            if (!has_marked_output)
                return false;

            auto convert_node = dynamic_pointer_cast<opset8::Convert>(node);
            if (convert_node){
                // if during propagating up there is a Convert it must go to Const,
                // otherwise interrupt propagation
                auto const_node = dynamic_pointer_cast<opset8::Constant>(node->input_value(0).get_node_shared_ptr());
                if (!const_node)
                    return false;
            }

            disable_fp16_compression(node);
            return true;
        };

        auto m = make_shared<pattern::Matcher>(propagate_through_ops, matcher_name);
        register_matcher(m, callback);
    }
};

class PropagateDownMarkToKeepInMixedPrecision : public pass::MatcherPass {
public:
    OPENVINO_RTTI("PropagateDownMarkToKeepInMixedPrecision", "0");
    PropagateDownMarkToKeepInMixedPrecision() {
        MATCHER_SCOPE(PropagateDownMarkToKeepInMixedPrecision);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            // on convert down propagation should be interrupted
            auto convert_node = dynamic_pointer_cast<opset8::Convert>(node);
            if (convert_node)
                return false;

            bool is_changed = false;
            for (const auto& in_node : node->input_values()) {
                if (!in_node.get_element_type().is_real())
                    continue;
                if (fp16_compression_is_disabled(in_node.get_node_shared_ptr())) {
                    disable_fp16_compression(node);
                    is_changed = true;
                    break;
                }
            }
            return is_changed;
        };
        auto m = make_shared<pattern::Matcher>(propagate_through_ops, matcher_name);
        register_matcher(m, callback);
    }
};

bool MarkSugraphsToKeepInMixedPrecision::run_on_model(const shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(MarkSugraphsToKeepInMixedPrecision);

    Manager manager(get_pass_config());
    REGISTER_PASS(manager, MarkExpReduceOpToKeepInMixedPrecision)
    REGISTER_PASS(manager, InitMarkToKeepInMixedPrecision)
    REGISTER_PASS(manager, MarkDivWithEpsToKeepInMixedPrecision)
    REGISTER_PASS(manager, PropagateDownMarkToKeepInMixedPrecision)

    auto propagate_up = manager.register_pass<BackwardGraphRewrite>();
    ADD_MATCHER(propagate_up, PropagateUpMarkToKeepInMixedPrecision)

    manager.run_passes(m);

    return false;  // no need to revalidate
}
}  // namespace pass
}  // namespace ov
