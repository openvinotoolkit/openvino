// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_subgraphs_to_keep_in_mixed_precision.hpp"

#include "itt.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset2.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/reduceop_path.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;

namespace ov {
namespace pass {

MarkNormalizationOps::MarkNormalizationOps() {
    MATCHER_SCOPE(MarkNormalizationOps);
    auto ops_to_be_kept_fp32 = pattern::wrap_type<opset2::MVN, opset10::MVN, opset10::NormalizeL2>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!node)
            return false;

        disable_fp16_compression(node);
        return true;
    };
    auto m = make_shared<pattern::Matcher>(ops_to_be_kept_fp32, matcher_name);
    register_matcher(m, callback);
}

std::shared_ptr<Node> propagate_through_ops =
    pattern::wrap_type<opset10::Squeeze,
                       opset10::Unsqueeze,
                       opset10::Reshape,
                       op::util::BroadcastBase,
                       op::util::BinaryElementwiseArithmetic,
                       op::util::UnaryElementwiseArithmetic,
                       opset10::MVN,
                       opset2::MVN,
                       opset10::NormalizeL2,
                       opset10::Sqrt,
                       opset10::StridedSlice,
                       opset10::ReduceSum,
                       opset10::ReduceMean,
                       opset10::Slice,
                       opset10::VariadicSplit,
                       opset10::Split,
                       op::util::GatherBase,
                       opset10::Concat,
                       opset10::Convert,  // through Convert can go only to Constants
                       opset10::Constant,
                       opset10::Tile>();

class PropagateUpMarkToKeepInMixedPrecision : public pass::MatcherPass {
public:
    OPENVINO_RTTI("PropagateUpMarkToKeepInMixedPrecision", "0");
    PropagateUpMarkToKeepInMixedPrecision() {
        MATCHER_SCOPE(PropagateUpMarkToKeepInMixedPrecision);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            bool has_marked_output = false;
            for (const auto& output : node->outputs()) {
                for (const auto& out_inputs : output.get_target_inputs()) {
                    if (out_inputs.get_element_type().is_real() &&
                        fp16_compression_is_disabled(out_inputs.get_node()->shared_from_this())) {
                        has_marked_output = true;
                    }
                }
            }

            if (!has_marked_output)
                return false;

            auto convert_node = dynamic_pointer_cast<opset10::Convert>(node);
            if (convert_node) {
                // if during propagating up there is a Convert it must go to Const,
                // otherwise interrupt propagation
                auto const_node = dynamic_pointer_cast<opset10::Constant>(node->input_value(0).get_node_shared_ptr());
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
            auto convert_node = dynamic_pointer_cast<opset10::Convert>(node);
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

    // Mark root of Division with eps pattern to keep in FP32
    REGISTER_PASS(manager, MarkDivWithEps)

    REGISTER_PASS(manager, MarkExpInReduceOpPath)
    REGISTER_PASS(manager, MarkNormalizationOps)
    REGISTER_PASS(manager, PropagateDownMarkToKeepInMixedPrecision)
    auto propagate_up = manager.register_pass<BackwardGraphRewrite>();
    ADD_MATCHER(propagate_up, PropagateUpMarkToKeepInMixedPrecision)

    // Mark nodes in ShapeOf subgraphs to keep in FP32
    REGISTER_PASS(manager, MarkPrecisionSensitiveShapeOfSubgraphs)

    manager.run_passes(m);

    return false;  // no need to revalidate
}

class InitMarkReduceOpPath : public pass::MatcherPass {
public:
    OPENVINO_RTTI("InitMarkReduceOpPath", "0");
    InitMarkReduceOpPath() {
        MATCHER_SCOPE(InitMarkReduceOpPath);

        auto reduce_ops = pattern::wrap_type<opset10::ReduceSum, opset10::ReduceMean>();

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

class PropagateMarkUpReduceOpPath : public pass::MatcherPass {
public:
    OPENVINO_RTTI("PropagateMarkUpReduceOpPath", "0");
    PropagateMarkUpReduceOpPath() {
        MATCHER_SCOPE(PropagateMarkUpReduceOpPath);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            for (const auto& output : node->outputs()) {
                for (const auto& out_inputs : output.get_target_inputs()) {
                    if (out_inputs.get_element_type().is_real() &&
                        is_reduceop_path(out_inputs.get_node()->shared_from_this())) {
                        mark_reduceop_path(node);
                        return false;
                    }
                }
            }
            return false;
        };
        auto m = make_shared<pattern::Matcher>(propagate_through_ops, matcher_name);
        register_matcher(m, callback);
    }
};

class MarkExp : public pass::MatcherPass {
public:
    // only exponent that go into ReduceOp should be marked as precision sensitive
    OPENVINO_RTTI("MarkExp", "0");
    MarkExp() {
        MATCHER_SCOPE(MarkExp);
        auto exp_pattern = pattern::wrap_type<opset10::Exp>();

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            if (!is_reduceop_path(node))
                return false;

            disable_fp16_compression(node);
            return true;
        };
        auto m = make_shared<pattern::Matcher>(exp_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

MarkExpInReduceOpPath::MarkExpInReduceOpPath() {
    // marking of ReduceOp path is needed to mark only Exponents that go into ReduceSum/ReduceMean
    ADD_MATCHER_FOR_THIS(InitMarkReduceOpPath);
    ADD_MATCHER_FOR_THIS(PropagateMarkUpReduceOpPath);
    ADD_MATCHER_FOR_THIS(MarkExp);
}

MarkDivWithEps::MarkDivWithEps() {
    MATCHER_SCOPE(MarkDivWithEps);

    // to detect the following patterns where eps is used to prevent division by zero:
    // input_1/Maximum(input_2, eps)
    // input_1/Add(input_2, eps)
    // input_1/Sqrt(Maximum(input_2, eps))
    // input_1/Sqrt(Add(input_2, eps))
    // input_1*Pow(Maximum(input_2, eps), -z)
    // input_1*Pow(Add(input_2, eps), -z)
    auto input_1 = pattern::any_input();
    auto input_2 = pattern::any_input();

    auto eps_const_pattern = pattern::wrap_type<opset10::Constant>();
    auto max = std::make_shared<opset10::Maximum>(input_2, eps_const_pattern);
    auto add = std::make_shared<opset10::Add>(input_2, eps_const_pattern);
    auto max_or_add = std::make_shared<pattern::op::Or>(OutputVector{max, add});

    auto sqrt = std::make_shared<opset10::Sqrt>(max_or_add);
    auto sqrt_or_max_add = std::make_shared<pattern::op::Or>(OutputVector{max_or_add, sqrt});
    // whether is divided directly or after sqrt (e.g. in L2Norm after sqrt, in MVN is divided directly)
    auto divide = std::make_shared<opset10::Divide>(input_1, sqrt_or_max_add);

    auto pow_exp = pattern::wrap_type<opset10::Constant>();
    auto convert_pattern = pattern::wrap_type<opset10::Convert>({pow_exp});
    auto pow_exp_or_convert = std::make_shared<pattern::op::Or>(OutputVector{pow_exp, convert_pattern});

    auto pow_pattern = std::make_shared<opset10::Power>(max_or_add, pow_exp_or_convert);
    auto mul_pattern = std::make_shared<opset10::Multiply>(input_1, pow_pattern);
    auto div_or_mul_to_negative_pow = std::make_shared<pattern::op::Or>(OutputVector{divide, mul_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        if (!m.get_match_root())
            return false;

        const auto mul = std::dynamic_pointer_cast<opset10::Multiply>(m.get_match_root());
        // if pattern input_1*Pow(Maximum(input_2, eps), z) or input_1*Pow(Add(input_2, eps), z) is matched
        // need to check that power is negative
        if (mul) {
            const auto pow_const = std::dynamic_pointer_cast<opset10::Constant>(pattern_to_output.at(pow_exp));
            if (pow_const) {
                // continue only if exponent is negative (z < 0)
                if (pow_const->get_element_type() == element::f16) {
                    for (auto val : pow_const->get_vector<float16>())
                        if (val >= 0.0f)
                            return false;
                } else if (pow_const->get_element_type() == element::f32) {
                    for (auto val : pow_const->get_vector<float>())
                        if (val >= 0.0f)
                            return false;
                }
            }
        }

        const auto eps_const = std::dynamic_pointer_cast<opset10::Constant>(pattern_to_output.at(eps_const_pattern));
        if (!eps_const)
            return false;
        if (eps_const->get_element_type() == element::f32) {
            for (const auto& val : eps_const->get_vector<float>())
                if (val > static_cast<float>(float16::from_bits(0x0400)))
                    return false;
        } else if (eps_const->get_element_type() == element::f16) {
            for (const auto& val : eps_const->get_vector<float16>())
                if (val > float16::from_bits(0x0400))
                    return false;
        }
        disable_fp16_compression(m.get_match_root());
        return true;
    };

    auto m = make_shared<pattern::Matcher>(div_or_mul_to_negative_pow, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
