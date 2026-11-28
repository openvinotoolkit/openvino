// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_subgraphs_to_keep_in_mixed_precision.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "itt.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/fp16_compression/mark_floatpoint_range.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

namespace ov::pass {

namespace {

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

void mark_fq_path(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace("fq_path", true);
}
bool is_fq_path(const std::shared_ptr<const Node>& node) {
    return node->get_rt_info().count("fq_path");
}

void erase_fq_path(const std::shared_ptr<Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase("fq_path");
}

// Marking continues to propagate through these ops.
const std::shared_ptr<Node> propagate_through_ops =
    pattern::wrap_type<v0::Squeeze,
                       v0::Unsqueeze,
                       v1::Reshape,
                       op::util::BroadcastBase,
                       op::util::BinaryElementwiseArithmetic,
                       op::util::UnaryElementwiseArithmetic,
                       ov::op::v6::MVN,
                       v0::MVN,
                       v0::NormalizeL2,
                       v0::Sqrt,
                       v1::StridedSlice,
                       v1::ReduceSum,
                       v1::ReduceMean,
                       v8::Slice,
                       v1::VariadicSplit,
                       v1::Split,
                       v0::Concat,
                       v0::Convert,  // through Convert can go only to Constants
                       v0::Constant,
                       v0::Tile,
                       v1::Select,
                       op::util::BinaryElementwiseComparison>();

// The same set of ops as in propagate_through_ops, plus logical ops so the walk can follow a
// boolean comparison result into a Select condition. Convert and Constant are handled explicitly
// while walking an integer-to-float subgraph.
bool is_precision_propagating_op(const std::shared_ptr<Node>& node) {
    return ov::is_type_any_of<v0::Squeeze,
                              v0::Unsqueeze,
                              v1::Reshape,
                              op::util::BroadcastBase,
                              op::util::BinaryElementwiseArithmetic,
                              op::util::UnaryElementwiseArithmetic,
                              ov::op::v6::MVN,
                              v0::MVN,
                              v0::NormalizeL2,
                              v0::Sqrt,
                              v1::StridedSlice,
                              v1::ReduceSum,
                              v1::ReduceMean,
                              v8::Slice,
                              v1::VariadicSplit,
                              v1::Split,
                              v0::Concat,
                              v0::Tile,
                              v1::Select,
                              op::util::BinaryElementwiseComparison,
                              op::util::BinaryElementwiseLogical,
                              v1::LogicalNot>(node);
}

}  // namespace

/* After PropagateDownMark we need to go also once up to include side branches of ops with several args:
 * Elementwise, Concat and so. E.g. if one of the argument of Concat was marked
 * during PropagateDownMark we need also mark its other inputs and this is done in this PropagateUpMark pass.
 * Propagation stops when we face ops not listed in propagate_through_ops: e.g. if we face Conv or MatMul.
 */
class PropagateUpMarkToKeepInMixedPrecision : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateUpMarkToKeepInMixedPrecision");
    PropagateUpMarkToKeepInMixedPrecision() {
        MATCHER_SCOPE(PropagateUpMarkToKeepInMixedPrecision);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            bool has_marked_output = false;
            for (const auto& output : node->outputs()) {
                for (const auto& out_inputs : output.get_target_inputs()) {
                    if (out_inputs.get_element_type().is_real() &&
                        is_conversion_disabled(out_inputs.get_node()->shared_from_this(), element::f16)) {
                        has_marked_output = true;
                    }
                }
            }

            if (!has_marked_output)
                return false;

            auto convert_node = as_type_ptr<v0::Convert>(node);
            if (convert_node) {
                // if during propagating up there is a Convert it must go to Const,
                // otherwise interrupt propagation
                auto const_node = as_type_ptr<v0::Constant>(node->input_value(0).get_node_shared_ptr());
                if (!const_node)
                    return false;
            }

            disable_conversion(node, element::f16);
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
class PropagateDownMarkToKeepInMixedPrecision : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateDownMarkToKeepInMixedPrecision");
    PropagateDownMarkToKeepInMixedPrecision() {
        MATCHER_SCOPE(PropagateDownMarkToKeepInMixedPrecision);

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            // on convert down propagation should be interrupted
            auto convert_node = as_type_ptr<v0::Convert>(node);
            if (convert_node)
                return false;

            bool is_changed = false;
            for (const auto& in_node : node->input_values()) {
                if (!in_node.get_element_type().is_real())
                    continue;
                if (is_fq_path(in_node.get_node_shared_ptr())) {
                    enable_conversion(node, element::f16);
                    return true;
                }

                if (is_conversion_disabled(in_node.get_node_shared_ptr(), element::f16)) {
                    disable_conversion(node, element::f16);
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

class InitMarkReduceOpPath : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("InitMarkReduceOpPath");
    InitMarkReduceOpPath() {
        MATCHER_SCOPE(InitMarkReduceOpPath);

        auto reduce_ops = pattern::wrap_type<v1::ReduceSum, v1::ReduceMean>();

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

class PropagateMarkUpReduceOpPath : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateMarkUpReduceOpPath");
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

class MarkExp : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkExp");
    // only exponent that go into ReduceOp should be marked as precision sensitive and kept in f32
    MarkExp() {
        MATCHER_SCOPE(MarkExp);
        auto exp_pattern = pattern::wrap_type<v0::Exp>();

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            if (!is_reduceop_path(node))
                return false;

            disable_conversion(node, element::f16);
            return true;
        };
        auto m = make_shared<pattern::Matcher>(exp_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

class MarkRandomUniform : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkRandomUniform");

    MarkRandomUniform() {
        MATCHER_SCOPE(MarkRandomUniform);
        auto random_uniform_pattern = pattern::wrap_type<v8::RandomUniform>();

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            disable_conversion(node, element::f16);
            for (const auto& output : node->outputs()) {
                auto target_inputs = output.get_target_inputs();
                for (const auto& input : target_inputs) {
                    auto consumer_node = input.get_node();
                    for (auto input : consumer_node->inputs()) {
                        mark_as_precision_sensitive(input);
                    }
                }
            }
            return false;
        };
        auto m = make_shared<pattern::Matcher>(random_uniform_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

/* MarkExpInReduceOpPath marks path that goes into ReduceSum and ReduceMean.
 * Values that go from Exp to ReduceSum/ReduceMean are precision
 * sensitive and should be kept in f32 precision for mixed inference.
 */
class MarkExpInReduceOpPath : public BackwardGraphRewrite {
public:
    OPENVINO_RTTI("MarkExpInReduceOpPath", "0", BackwardGraphRewrite);
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
    OPENVINO_MATCHER_PASS_RTTI("MarkDivWithEps");
    MarkDivWithEps() {
        MATCHER_SCOPE(MarkDivWithEps);

        // to detect the following patterns where eps is used to prevent division by zero:
        // input_1 / Maximum(input_2, eps)
        // input_1 / Add(input_2, eps)
        // input_1 / Sqrt(Maximum(input_2, eps))
        // input_1 / Sqrt(Add(input_2, eps))
        // input_1 * Pow(Maximum(input_2, eps), -z)
        // input_1 * Pow(Add(input_2, eps), -z)

        auto input_1 = pattern::any_input();
        auto input_2 = pattern::any_input();

        auto eps_const_pattern = pattern::wrap_type<v0::Constant>();
        auto optional_eps_convert = pattern::optional<v0::Convert>(eps_const_pattern);

        auto max_or_add = pattern::wrap_type<v1::Maximum, v1::Add>(OutputVector{input_2, optional_eps_convert});

        auto optional_sqrt = pattern::optional<v0::Sqrt>(max_or_add);
        // whether is divided directly or after sqrt (e.g. in L2Norm after sqrt, in MVN is divided directly)
        auto divide = std::make_shared<v1::Divide>(input_1, optional_sqrt);

        auto pow_exp = pattern::wrap_type<v0::Constant>();
        auto optional_pow_convert = pattern::optional<v0::Convert>(pow_exp);

        auto pow_pattern = std::make_shared<v1::Power>(max_or_add, optional_pow_convert);
        auto mul_pattern = std::make_shared<v1::Multiply>(input_1, pow_pattern);
        auto div_or_mul_to_negative_pow = std::make_shared<pattern::op::Or>(OutputVector{divide, mul_pattern});

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& pattern_to_output = m.get_pattern_map();
            if (!m.get_match_root())
                return false;

            const auto mul = as_type_ptr<v1::Multiply>(m.get_match_root());
            // if pattern input_1*Pow(Maximum(input_2, eps), z) or input_1*Pow(Add(input_2, eps), z) is matched
            // need to check that power is negative
            if (mul) {
                const auto pow_const = as_type_ptr<v0::Constant>(pattern_to_output.at(pow_exp));
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

            const auto eps_const = as_type_ptr<v0::Constant>(pattern_to_output.at(eps_const_pattern));
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
            disable_conversion(m.get_match_root(), element::f16);
            return true;
        };

        auto m = make_shared<pattern::Matcher>(div_or_mul_to_negative_pow, matcher_name);
        register_matcher(m, callback);
    }
};

/* An integer-to-float subgraph that is converted back to an integer is precision sensitive:
 * f16 cannot represent every f32 value, and rounding before an operation like Ceiling can change
 * the resulting integer. A typical example is the DeBERTa relative position bucket computation,
 * where a wrong bucket silently changes the gathered position embedding.
 *
 * Mark the input of the float-to-integer Convert as precision sensitive, so that the backward
 * elementwise chain up to the integer-to-float Convert is kept in f32 by
 * MarkPrecisionSensitiveShapeOfSubgraphs.
 */
class MarkIntegerToFloatSubgraph : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkIntegerToFloatSubgraph");
    MarkIntegerToFloatSubgraph() {
        MATCHER_SCOPE(MarkIntegerToFloatSubgraph);
        auto convert_pattern = pattern::wrap_type<v0::Convert>();

        // Several integer-to-float Converts can feed the same bucket chain. Cache the terminal
        // conversions reachable from each node so the shared tail is walked only once per pass
        // run. The pass changes only rt_info, so a match cannot invalidate cached results.
        using TerminalInput = std::pair<Node*, size_t>;
        auto terminals_cache = std::make_shared<std::unordered_map<const Node*, std::vector<TerminalInput>>>();
        std::function<std::vector<TerminalInput>(const Node*)> find_terminals = [terminals_cache](const Node* node) {
            if (const auto cached = terminals_cache->find(node); cached != terminals_cache->end())
                return cached->second;

            std::vector<TerminalInput> terminals;
            std::unordered_set<const Node*> visited{node};
            std::vector<const Node*> nodes{node};
            while (!nodes.empty()) {
                const auto current = nodes.back();
                nodes.pop_back();
                if (const auto cached = terminals_cache->find(current); cached != terminals_cache->end()) {
                    terminals.insert(terminals.end(), cached->second.begin(), cached->second.end());
                    continue;
                }
                for (const auto& output : current->outputs()) {
                    for (const auto& target_input : output.get_target_inputs()) {
                        const auto consumer = target_input.get_node();
                        if (!visited.insert(consumer).second)
                            continue;

                        if (const auto to_integer = as_type_ptr<v0::Convert>(consumer->shared_from_this())) {
                            const auto terminal_input_type = to_integer->get_input_element_type(0);
                            // A boolean terminal is a comparison result converted to an index; the
                            // comparison itself is still computed from the floating point chain, so
                            // the chain must stay in f32 as well.
                            if ((terminal_input_type.is_real() || terminal_input_type == element::boolean) &&
                                to_integer->get_convert_element_type().is_integral_number())
                                terminals.emplace_back(consumer, target_input.get_index());
                            // Any other Convert interrupts the chain, the same way it interrupts
                            // down propagation.
                            continue;
                        }

                        if (!is_precision_propagating_op(consumer->shared_from_this()))
                            continue;

                        // Follow every output, not just floating point ones: a comparison produces
                        // a boolean condition, but f16 rounding of the compared values can still
                        // change which branch a following Select takes and hence the final integer.
                        nodes.push_back(consumer);
                    }
                }
            }
            terminals_cache->emplace(node, std::move(terminals));
            return terminals_cache->at(node);
        };

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto convert = as_type_ptr<v0::Convert>(m.get_match_root());
            if (!convert)
                return false;
            if (!convert->get_input_element_type(0).is_integral_number() ||
                !convert->get_convert_element_type().is_real())
                return false;

            const auto terminals = find_terminals(convert.get());
            if (terminals.empty())
                return false;
            // Marking the input instead of the Convert itself keeps only the backward chain in f32:
            // MarkPrecisionSensitiveShapeOfSubgraphs walks it and disables f16 conversion, while a
            // mark on the source Convert would propagate down into unrelated forward branches.
            for (const auto& terminal : terminals)
                mark_as_precision_sensitive(terminal.first->input(terminal.second));
            return true;
        };

        auto m = make_shared<pattern::Matcher>(convert_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

class PropagateDownDisableSensitivityForQuantized : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PropagateDownDisableSensitivityForQuantized");
    PropagateDownDisableSensitivityForQuantized() {
        MATCHER_SCOPE(PropagateDownDisableSensitivityForQuantized);

        // through this nodes
        const std::shared_ptr<Node> quantization_propagating_nodes = pattern::wrap_type<v0::Squeeze,
                                                                                        v0::Unsqueeze,
                                                                                        v0::FakeQuantize,
                                                                                        v1::Reshape,
                                                                                        op::util::BroadcastBase,
                                                                                        v0::DepthToSpace,
                                                                                        v0::Interpolate,
                                                                                        ov::op::v4::Interpolate,
                                                                                        ov::op::v11::Interpolate,
                                                                                        v1::MaxPool,
                                                                                        v8::MaxPool,
                                                                                        op::util::PadBase,
                                                                                        v1::ReduceMax,
                                                                                        v1::ReduceMin,
                                                                                        v0::Relu,
                                                                                        v1::Transpose,
                                                                                        v0::ShuffleChannels,
                                                                                        v1::StridedSlice,
                                                                                        v8::Slice,
                                                                                        v1::VariadicSplit,
                                                                                        v1::Split,
                                                                                        v0::Concat,
                                                                                        v0::Tile>();

        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& node = m.get_match_root();
            if (!node)
                return false;

            auto is_quantize = as_type_ptr<v0::FakeQuantize>(node);
            if (is_quantize) {
                mark_fq_path(node);
                return true;
            }

            bool is_changed = false;

            for (const auto& in_node_output : node->input_values()) {
                auto input_node = in_node_output.get_node_shared_ptr();
                auto is_quantize = as_type_ptr<v0::FakeQuantize>(input_node);
                if (is_quantize || is_fq_path(input_node)) {
                    mark_fq_path(node);
                    enable_conversion(node, element::f16);
                    is_changed = true;
                }
            }

            return is_changed;
        };
        auto m = make_shared<pattern::Matcher>(quantization_propagating_nodes, matcher_name);
        register_matcher(m, callback);
    }
};

bool MarkSugraphsToKeepInMixedPrecision::run_on_model(const shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(MarkSugraphsToKeepInMixedPrecision);

    Manager manager(get_pass_config(), "MarkSugraphsToKeepInMixedPrecision");
    manager.set_per_pass_validation(false);
    // Mark root of Division with eps pattern to keep in FP32
    REGISTER_PASS(manager, MarkFloatingPointRange)
    REGISTER_PASS(manager, MarkDivWithEps)
    REGISTER_PASS(manager, MarkExpInReduceOpPath)
    REGISTER_PASS(manager, MarkRandomUniform)
    REGISTER_PASS(manager, PropagateDownDisableSensitivityForQuantized)
    // Mark integer-to-float subgraphs that are converted back to integers to keep them in FP32
    REGISTER_PASS(manager, MarkIntegerToFloatSubgraph)

    // both Up and Down propagations are needed.
    // Why both of them are needed is explained in comments in passes declarations.
    REGISTER_PASS(manager, PropagateDownMarkToKeepInMixedPrecision)

    auto propagate_up = manager.register_pass<BackwardGraphRewrite>();
    ADD_MATCHER(propagate_up, PropagateUpMarkToKeepInMixedPrecision)

    // Mark nodes in ShapeOf subgraphs to keep in FP32
    REGISTER_PASS(manager, MarkPrecisionSensitiveShapeOfSubgraphs)
    manager.run_passes(m);

    for (auto& node : m->get_ops()) {
        erase_reduceop_path(node);
        erase_fq_path(node);
        erase_range_path(node);
    }

    return false;  // no need to revalidate
}

}  // namespace ov::pass
