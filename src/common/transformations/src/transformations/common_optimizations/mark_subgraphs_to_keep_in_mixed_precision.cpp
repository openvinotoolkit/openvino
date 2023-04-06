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
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov::opset10;

namespace ov {
namespace pass {

/*
 * MarkNormalizationOps marks MVN and NormalizeL2 to be kept in f32 precision.
 */
class MarkNormalizationOps : public MatcherPass {
public:
    OPENVINO_RTTI("MarkNormalizationOps", "0");

    MarkNormalizationOps() {
        MATCHER_SCOPE(MarkNormalizationOps);
        auto ops_to_be_kept_fp32 = pattern::wrap_type<opset2::MVN, MVN, NormalizeL2>();

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
};

// Marking continues to propagate through these ops.
std::shared_ptr<Node> propagate_through_ops = pattern::wrap_type<Squeeze,
                                                                 Unsqueeze,
                                                                 Reshape,
                                                                 op::util::BroadcastBase,
                                                                 op::util::BinaryElementwiseArithmetic,
                                                                 op::util::UnaryElementwiseArithmetic,
                                                                 MVN,
                                                                 opset2::MVN,
                                                                 NormalizeL2,
                                                                 Sqrt,
                                                                 StridedSlice,
                                                                 ReduceSum,
                                                                 ReduceMean,
                                                                 Slice,
                                                                 VariadicSplit,
                                                                 Split,
                                                                 op::util::GatherBase,
                                                                 Concat,
                                                                 Convert,  // through Convert can go only to Constants
                                                                 Constant,
                                                                 Tile>();

/* After PropagateDownMark we need to go also once up to include side branches of ops with several args:
 * Elementwise, Concat and so. E.g. if one of the argument of Concat was marked
 * during PropagateDownMark we need also mark its other inputs and this is done in this PropagateUpMark pass.
 * Propagation stops when we face ops not listed in propagate_through_ops: e.g. if we face Conv or MatMul.
 */
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

            auto convert_node = dynamic_pointer_cast<Convert>(node);
            if (convert_node) {
                // if during propagating up there is a Convert it must go to Const,
                // otherwise interrupt propagation
                auto const_node = dynamic_pointer_cast<Constant>(node->input_value(0).get_node_shared_ptr());
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

/* Starting from the marked precision sensitive nodes we need to propagate down to include neighboring
 * ops like Slice, ReduceSum, Reshape, Elementwise, et al. to be kept in f32 as well.
 * Propagation stops when ops not listed in propagate_through_ops are faced: e.g. if we face Conv or MatMul.
 */
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
            auto convert_node = dynamic_pointer_cast<Convert>(node);
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

void mark_reduceop_path(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace("reduceop_path", true);
}
bool is_reduceop_path(const std::shared_ptr<const Node>& node) {
    return node->get_rt_info().count("reduceop_path");
}

void erase_reduceop_path(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase("reduceop_path");
}

class InitMarkReduceOpPath : public pass::MatcherPass {
public:
    OPENVINO_RTTI("InitMarkReduceOpPath", "0");
    InitMarkReduceOpPath() {
        MATCHER_SCOPE(InitMarkReduceOpPath);

        auto reduce_ops = pattern::wrap_type<ReduceSum, ReduceMean>();

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
    // only exponent that go into ReduceOp should be marked as precision sensitive and kept in f32
    OPENVINO_RTTI("MarkExp", "0");
    MarkExp() {
        MATCHER_SCOPE(MarkExp);
        auto exp_pattern = pattern::wrap_type<Exp>();

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
/* MarkExpInReduceOpPath marks path that goes into ReduceSum and ReduceMean.
 * Values that go from Exp to ReduceSum/ReduceMean are precision
 * sensitive and should be kept in f32 precision for mixed inference.
 */
class MarkExpInReduceOpPath : public BackwardGraphRewrite {
public:
    OPENVINO_RTTI("MarkExpInReduceOpPath", "0");
    MarkExpInReduceOpPath() {
        // marking of ReduceOp path is needed to mark only Exponents that go into ReduceSum/ReduceMean
        ADD_MATCHER_FOR_THIS(InitMarkReduceOpPath);
        ADD_MATCHER_FOR_THIS(PropagateMarkUpReduceOpPath);
        ADD_MATCHER_FOR_THIS(MarkExp);
    }
};

/* MarkDivWithEps martk pattern that matches the patterns input_1/Maximum(input_2, eps); input_1/Add(input_2, eps);
 * and input_1*Pow(Maximum[Add](input_2, eps), -z) and marks subgraph root to be kept in fp32.
 *
 * If both input_1 and input_2 simultaneously happen to be zero to prevent from NaNs and not to loose accuracy,
 * we should calculate such patterns always in fp32 precision even if ov::Model is compressed to fp16.
 */
class MarkDivWithEps : public MatcherPass {
public:
    OPENVINO_RTTI("MarkDivWithEps", "0");
    MarkDivWithEps() {
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

        auto eps_const_pattern = pattern::wrap_type<Constant>();
        auto max_or_add = pattern::wrap_type<Maximum, Add>(OutputVector{input_2, eps_const_pattern});

        auto sqrt = std::make_shared<Sqrt>(max_or_add);
        auto sqrt_or_max_add = std::make_shared<pattern::op::Or>(OutputVector{max_or_add, sqrt});
        // whether is divided directly or after sqrt (e.g. in L2Norm after sqrt, in MVN is divided directly)
        auto divide = std::make_shared<Divide>(input_1, sqrt_or_max_add);

        auto pow_exp = pattern::wrap_type<Constant>();
        auto convert_pattern = pattern::wrap_type<Convert>({pow_exp});
        auto pow_exp_or_convert = std::make_shared<pattern::op::Or>(OutputVector{pow_exp, convert_pattern});

        auto pow_pattern = std::make_shared<Power>(max_or_add, pow_exp_or_convert);
        auto mul_pattern = std::make_shared<Multiply>(input_1, pow_pattern);
        auto div_or_mul_to_negative_pow = std::make_shared<pattern::op::Or>(OutputVector{divide, mul_pattern});

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& pattern_to_output = m.get_pattern_map();
            if (!m.get_match_root())
                return false;

            const auto mul = std::dynamic_pointer_cast<Multiply>(m.get_match_root());
            // if pattern input_1*Pow(Maximum(input_2, eps), z) or input_1*Pow(Add(input_2, eps), z) is matched
            // need to check that power is negative
            if (mul) {
                const auto pow_const = std::dynamic_pointer_cast<Constant>(pattern_to_output.at(pow_exp));
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

            const auto eps_const = std::dynamic_pointer_cast<Constant>(pattern_to_output.at(eps_const_pattern));
            if (!eps_const)
                return false;
            if (eps_const->get_element_type() == element::f32) {
                for (const auto& val : eps_const->get_vector<float>())
                    if (val > static_cast<float>(float16_min_normalized))
                        return false;
            } else if (eps_const->get_element_type() == element::f16) {
                for (const auto& val : eps_const->get_vector<float16>())
                    if (val > float16_min_normalized)
                        return false;
            }
            disable_fp16_compression(m.get_match_root());
            return true;
        };

        auto m = make_shared<pattern::Matcher>(div_or_mul_to_negative_pow, matcher_name);
        register_matcher(m, callback);
    }
};

bool MarkSugraphsToKeepInMixedPrecision::run_on_model(const shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(MarkSugraphsToKeepInMixedPrecision);

    Manager manager(get_pass_config());
    // Mark root of Division with eps pattern to keep in FP32
    REGISTER_PASS(manager, MarkDivWithEps)
    REGISTER_PASS(manager, MarkExpInReduceOpPath)

    // both Up and Down propagations are needed.
    // Why both of them are needed is explained in comments in passes declarations.
    REGISTER_PASS(manager, PropagateDownMarkToKeepInMixedPrecision)

    auto propagate_up = manager.register_pass<BackwardGraphRewrite>();
    ADD_MATCHER(propagate_up, PropagateUpMarkToKeepInMixedPrecision)

    // Mark nodes in ShapeOf subgraphs to keep in FP32
    REGISTER_PASS(manager, MarkPrecisionSensitiveShapeOfSubgraphs)
    REGISTER_PASS(manager, MarkNormalizationOps)
    manager.run_passes(m);

    for (auto& node : m->get_ops()) {
        erase_reduceop_path(node);
    }

    return false;  // no need to revalidate
}

}  // namespace pass
}  // namespace ov
