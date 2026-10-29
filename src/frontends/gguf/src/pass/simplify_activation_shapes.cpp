// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/simplify_activation_shapes.hpp"

#include <algorithm>
#include <unordered_set>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/symbolic_info.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::frontend::gguf::pass {
namespace {

bool activation_shape(const ov::PartialShape& shape) {
    return shape.rank() == 4 && shape[1] == 1;
}

ov::PartialShape reduced_shape(const ov::PartialShape& shape) {
    return {shape[0], shape[2], shape[3]};
}

bool same_shape(const ov::PartialShape& lhs, const ov::PartialShape& rhs) {
    if (lhs.rank().is_dynamic() || lhs.rank() != rhs.rank())
        return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!ov::symbol::util::dims_are_equal(lhs[i], rhs[i]))
            return false;
    }
    return true;
}

ov::Output<ov::Node> unwrapped(const ov::Output<ov::Node>& value) {
    const auto node = ov::as_type_ptr<ov::op::v0::Unsqueeze>(value.get_node_shared_ptr());
    if (node && node->input_value(0).get_partial_shape().rank() == 3 &&
        ov::op::util::has_constant_value<int64_t>(node->get_input_node_shared_ptr(1), {1}))
        return node->input_value(0);
    return value;
}

}  // namespace

bool SimplifyActivationShapes::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager symbols;
    symbols.set_per_pass_validation(false);
    symbols.register_pass<ov::pass::SymbolicPropagation>();
    symbols.run_passes(model);
    const auto nodes = model->get_ordered_ops();
    std::unordered_set<const ov::Node*> constants;
    for (const auto& node : nodes) {
        const auto inputs = node->inputs();
        if (ov::is_type<ov::op::v0::Constant>(node) ||
            (!inputs.empty() && std::all_of(inputs.begin(), inputs.end(), [&](const auto& input) {
                return constants.count(input.get_source_output().get_node());
            }))) {
            constants.insert(node.get());
        }
    }
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    bool changed = false;
    for (const auto& node : nodes) {
        if (constants.count(node.get()) || node->get_output_size() != 1 ||
            !node->get_output_element_type(0).is_real() || transformation_callback(node)) {
            continue;
        }
        const auto shape = node->get_output_partial_shape(0);
        const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(node);
        if (reshape && shape.rank() == 3) {
            const auto value = unwrapped(reshape->input_value(0));
            if (same_shape(value.get_partial_shape(), shape)) {
                changed |= ov::replace_output_update_name(node->output(0), value);
            }
            continue;
        }
        if (!activation_shape(shape))
            continue;
        ov::Output<ov::Node> result;
        ov::NodeVector created;
        const auto lower = [&](const ov::Output<ov::Node>& input) -> ov::Output<ov::Node> {
            if (!activation_shape(input.get_partial_shape()))
                return input;
            const auto value = unwrapped(input);
            if (value != input)
                return value;
            const auto squeezed = ov::op::util::make_try_fold<ov::op::v0::Squeeze>(input, axis);
            created.push_back(squeezed);
            return squeezed;
        };
        if (reshape) {
            const auto value = unwrapped(reshape->input_value(0));
            if (same_shape(value.get_partial_shape(), reduced_shape(shape))) {
                result = value;
            } else {
                const bool reduced_input = activation_shape(reshape->get_input_partial_shape(0));
                if (reshape->get_special_zero() && !reduced_input)
                    continue;
                const auto indices = ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 2, 3});
                const auto zero = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
                const auto target =
                    ov::op::util::make_try_fold<ov::op::v8::Gather>(reshape->input_value(1), indices, zero);
                created.insert(created.end(), {indices, zero, target});
                result = std::make_shared<ov::op::v1::Reshape>(reduced_input ? lower(reshape->input_value(0)) : value,
                                                               target,
                                                               reshape->get_special_zero());
            }
        } else if (ov::is_type_any_of<ov::op::v1::ReduceMean, ov::op::v1::ReduceSum, ov::op::v0::NormalizeL2>(node)) {
            if (!activation_shape(node->get_input_partial_shape(0)))
                continue;
            const auto axes = ov::as_type_ptr<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
            if (!ov::op::util::has_constant_value<int64_t>(axes, {-1}) &&
                !ov::op::util::has_constant_value<int64_t>(axes, {3}))
                continue;
            const auto last = ov::op::v0::Constant::create(axes->get_element_type(), axes->get_shape(), {-1});
            created.push_back(last);
            result = node->clone_with_new_inputs({lower(node->input_value(0)), last});
        } else {
            const auto binary = ov::as_type_ptr<ov::op::util::BinaryElementwiseArithmetic>(node);
            if (binary) {
                const auto type = binary->get_autob().m_type;
                if (type != ov::op::AutoBroadcastType::NUMPY && type != ov::op::AutoBroadcastType::NONE)
                    continue;
                const auto inputs = node->input_values();
                if (type == ov::op::AutoBroadcastType::NONE &&
                    !std::all_of(inputs.begin(), inputs.end(), [](const auto& input) {
                        return activation_shape(input.get_partial_shape());
                    }))
                    continue;
            } else if (!ov::is_type_any_of<ov::op::util::UnaryElementwiseArithmetic,
                                           ov::op::v0::Convert,
                                           ov::op::v4::Swish>(node)) {
                continue;
            }
            ov::OutputVector inputs;
            for (const auto& input : node->input_values())
                inputs.push_back(lower(input));
            result = node->clone_with_new_inputs(inputs);
        }
        const auto restored = std::make_shared<ov::op::v0::Unsqueeze>(result, axis);
        restored->set_friendly_name(node->get_friendly_name());
        created.insert(created.end(), {result.get_node_shared_ptr(), restored});
        ov::copy_runtime_info({node, result.get_node_shared_ptr()}, created);
        ov::replace_node(node, restored);
        changed = true;
    }
    ov::remove_skip_invalidation_rti(model);
    return changed;
}

}  // namespace ov::frontend::gguf::pass
