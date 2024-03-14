// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"

#include <algorithm>
#include <numeric>

#include "bound_evaluate.hpp"
#include "compare.hpp"
#include "openvino/core/constant_fold_utils.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/util/common_util.hpp"
#include "sequnce_generator.hpp"

namespace {
const auto normalize_axis_to = [](const int64_t& tensor_rank) {
    return [&tensor_rank](int64_t& axis) {
        if (axis < 0) {
            axis += tensor_rank;
        }
    };
};

std::string normalize_axis_error_msg(const int64_t& axis, const int64_t& lower, const int64_t& upper) {
    return std::string(" Parameter axis ")
        .append(std::to_string(axis))
        .append(" out of the tensor rank range [")
        .append(std::to_string(lower))
        .append(", ")
        .append(std::to_string(upper))
        .append("].");
}

ov::OutputVector get_inputs_from_map(const std::shared_ptr<ov::Node>& node,
                                     const std::map<ov::Output<ov::Node>, std::shared_ptr<ov::Node>>& node_map) {
    size_t num_inputs = node->get_input_size();

    ov::OutputVector inputs;
    inputs.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; i++) {
        auto input = node->input_value(i);
        if (node_map.count(input) > 0) {
            inputs.push_back(node_map.at(input));
        } else {
            inputs.push_back(input);
        }
    }

    return inputs;
}

}  // namespace

int64_t ov::util::normalize(const int64_t& value, const int64_t& max) {
    return (value < 0) ? value + max : value;
};

bool ov::util::are_unique(const std::vector<int64_t>& data) {
    return std::unordered_set<int64_t>(data.begin(), data.cend()).size() == data.size();
}

// clip value to min, max
int64_t ov::util::clip(const int64_t& value, const int64_t& min, const int64_t& max) {
    return std::min(std::max(value, min), max);
};

std::shared_ptr<ov::op::v0::Constant> ov::util::constantfold_subgraph(const ov::Output<ov::Node>& subgraph_sink) {
    if (const auto& c = ov::as_type_ptr<op::v0::Constant>(subgraph_sink.get_node_shared_ptr()))
        return c;

    std::map<Output<Node>, std::shared_ptr<Node>> node_map;
    std::stack<std::shared_ptr<Node>> stack;
    stack.push(subgraph_sink.get_node_shared_ptr());

    while (!stack.empty()) {
        auto node = stack.top();
        size_t num_inputs = node->get_input_size();
        if (num_inputs == 0)
            return nullptr;

        if (ov::pass::constant_folding_is_disabled(node))
            return nullptr;

        if (ov::is_type<op::util::ShapeOfBase>(node) && node->get_input_partial_shape(0).is_dynamic())
            return nullptr;

        bool node_has_bounds_set = true;
        for (size_t i = 0; i < node->get_output_size(); i++) {
            const auto& tensor = node->get_output_tensor(i);
            bool tensor_has_bounds_set = tensor.has_and_set_bound();
            node_has_bounds_set = node_has_bounds_set && tensor_has_bounds_set;
            if (tensor_has_bounds_set) {
                const auto& lower = node->get_output_tensor(i).get_lower_value();
                auto constant = std::make_shared<ov::op::v0::Constant>(lower);
                node_map[node->output(i)] = constant;
            }
        }

        if (node_has_bounds_set) {
            stack.pop();
            continue;
        }

        auto original_node = node;
        const auto inputs = get_inputs_from_map(node, node_map);

        if (ov::util::node_requires_precision_conversion(node.get())) {
            node = ov::util::convert_to_supported_precision(node.get(), inputs);
        }

        OutputVector outputs(node->get_output_size());
        if (node->constant_fold(outputs, inputs)) {
            stack.pop();
            for (size_t i = 0; i < outputs.size(); i++) {
                node_map[original_node->output(i)] = outputs[i].get_node_shared_ptr();
            }
        } else {
            size_t stack_size_before = stack.size();
            for (size_t i = node->get_input_size(); i > 0; i--) {
                auto input = original_node->input_value(i - 1);
                if (node_map.count(input) == 0 && !ov::op::util::is_constant(input.get_node())) {
                    stack.push(input.get_node_shared_ptr());
                }
            }
            // if none of the inputs was pushed to stack, it means the node that was not constantfolded
            // is processed the second time. If that case - the node is not constfoldable.
            // A good example would be a node that all of its inputs are constants and yet it cannot be constantfolded
            // for some reason (like lack of evaluate, it's a op::util::FrameworkNode, etc.).
            if (stack_size_before == stack.size()) {
                return nullptr;
            }
        }
    }

    auto constant = node_map.at(subgraph_sink);
    if (constant->get_element_type() != subgraph_sink.get_element_type()) {
        auto convert = std::make_shared<op::v0::Convert>(constant, subgraph_sink.get_element_type());
        OutputVector output(1);
        if (!convert->constant_fold(output, OutputVector{constant}))
            return nullptr;
        constant = output[0].get_node_shared_ptr();
    }

    return ov::as_type_ptr<op::v0::Constant>(constant);
}

namespace ov {
namespace util {
using ov::op::v0::Constant;

std::shared_ptr<Constant> get_constant_from_source(const ov::Output<ov::Node>& source) {
    if (const auto& c = ov::as_type_ptr<Constant>(source.get_node_shared_ptr())) {
        return c;
    } else if (has_and_set_equal_bounds(source)) {
        return std::make_shared<Constant>(source.get_tensor().get_upper_value());
    } else {
        return {};
    }
}

template <class T>
Tensor make_tensor_of_max_value(const element::Type_t et) {
    Tensor t{et, Shape{}};
    *t.data<T>() = std::numeric_limits<T>::max();
    return t;
}

Tensor make_tensor_of_max_value(const element::Type_t et) {
    switch (et) {
    case element::boolean:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::boolean>>(et);
    case element::bf16:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::bf16>>(et);
    case element::f16:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::f16>>(et);
    case element::f32:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::f32>>(et);
    case element::f64:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::f64>>(et);
    case element::i8:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::i8>>(et);
    case element::i16:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::i16>>(et);
    case element::i32:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::i32>>(et);
    case element::i64:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::i64>>(et);
    case element::u1:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u1>>(et);
    case element::u8:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u8>>(et);
    case element::u16:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u16>>(et);
    case element::u32:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u32>>(et);
    case element::u64:
        return make_tensor_of_max_value<ov::fundamental_type_for<element::u64>>(et);
    default:
        return {};
    }
}

template <class T>
Tensor make_tensor_of_min_value(const element::Type_t et) {
    Tensor t{et, Shape{}};
    *t.data<T>() = std::numeric_limits<T>::min();
    return t;
}

Tensor make_tensor_of_min_value(const element::Type_t et) {
    switch (et) {
    case element::boolean:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::boolean>>(et);
    case element::bf16:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::bf16>>(et);
    case element::f16:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::f16>>(et);
    case element::f32:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::f32>>(et);
    case element::f64:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::f64>>(et);
    case element::i8:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::i8>>(et);
    case element::i16:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::i16>>(et);
    case element::i32:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::i32>>(et);
    case element::i64:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::i64>>(et);
    case element::u1:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u1>>(et);
    case element::u8:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u8>>(et);
    case element::u16:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u16>>(et);
    case element::u32:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u32>>(et);
    case element::u64:
        return make_tensor_of_min_value<ov::fundamental_type_for<element::u64>>(et);
    default:
        return {};
    }
}

std::vector<ov::PartialShape> get_tensors_partial_shapes(const TensorVector& tensors) {
    std::vector<ov::PartialShape> shapes;
    shapes.reserve(tensors.size());
    for (const auto& t : tensors) {
        shapes.emplace_back(t.get_shape());
    }
    return shapes;
}

std::vector<ov::PartialShape> get_node_input_partial_shapes(const ov::Node& node) {
    std::vector<ov::PartialShape> shapes;
    shapes.reserve(node.get_input_size());
    for (size_t i = 0; i < node.get_input_size(); ++i) {
        shapes.push_back(node.get_input_partial_shape(i));
    }
    return shapes;
}

bool is_rank_compatible_any_of(const Rank& r, std::initializer_list<Rank> others) {
    return std::any_of(others.begin(), others.end(), [&r](const Rank& other) {
        return r.compatible(other);
    });
}

bool evaluate_as_partial_shape(const ov::Output<ov::Node>& output, ov::PartialShape& pshape) {
    Tensor lb, ub;
    std::tie(lb, ub) = evaluate_both_bounds(output);
    bool shape_defined = false;
    if (lb && ub) {
        auto lower_bound = std::make_shared<op::v0::Constant>(lb.get_element_type(), lb.get_shape(), lb.data())
                               ->cast_vector<int64_t>();
        auto upper_bound = std::make_shared<op::v0::Constant>(ub.get_element_type(), ub.get_shape(), ub.data())
                               ->cast_vector<int64_t>();
        OPENVINO_ASSERT(lower_bound.size() == upper_bound.size());
        const TensorLabel& labels = output.get_tensor().get_value_label();
        OPENVINO_ASSERT(labels.empty() || lower_bound.size() == labels.size());

        std::vector<Dimension> resulting_pshape(lower_bound.size());
        for (size_t i = 0; i < lower_bound.size(); ++i) {
            auto low = lower_bound[i], up = upper_bound[i];
            OPENVINO_ASSERT(low >= 0 && up >= 0, "Value for partial shape evaluation can't be lower than zero.");
            if (output.get_element_type() == element::i32 && low != up) {
                if (up == std::numeric_limits<std::int32_t>::max())
                    up = std::numeric_limits<std::int64_t>::max();
                if (low == std::numeric_limits<std::int32_t>::max())
                    low = std::numeric_limits<std::int64_t>::max();
            }
            resulting_pshape[i] = {low, up};
            if (!labels.empty() && labels[i])
                DimensionTracker::set_label(resulting_pshape[i], labels[i]);
        }
        pshape = ov::PartialShape(resulting_pshape);
        shape_defined = true;
    }
    return shape_defined;
}

bool default_label_evaluator(const ov::Node* node, TensorLabelVector& output_labels) {
    return default_label_evaluator(node, {0}, output_labels);
}

void generate_transpose_default_order(std::vector<int64_t>& axes_order, const size_t length) {
    axes_order.reserve(axes_order.size() + length);
    std::generate_n(std::back_inserter(axes_order), length, ov::SeqGen<size_t, ov::Direction::BACKWARD>(length - 1));
}

bool is_valid_axes_order(const std::vector<int64_t>& axes_order, const size_t size) {
    return are_unique(axes_order) &&
           std::all_of(axes_order.cbegin(), axes_order.cend(), ov::cmp::Between<int64_t, ov::cmp::LOWER>(0, size));
}

bool has_no_labels(const ov::TensorLabel& labels) {
    return std::all_of(labels.cbegin(), labels.cend(), cmp::Equal<size_t>(no_label));
}

std::vector<size_t> normalize_axes(const std::string& node_description,
                                   const std::vector<int64_t>& axes,
                                   const Rank& tensor_rank) {
    std::vector<size_t> new_axes;
    new_axes.reserve(axes.size());
    for (const auto& axis : axes) {
        new_axes.push_back(ov::util::normalize_axis(node_description, axis, tensor_rank));
    }
    return new_axes;
}

void normalize_axes(const ov::Node* node, const int64_t& tensor_rank, std::vector<int64_t>& axes) {
    const auto axis_checker = cmp::Between<int64_t, cmp::BOTH>(-tensor_rank, tensor_rank ? (tensor_rank - 1) : 0);
    const auto invalid_axis = std::find_if_not(axes.cbegin(), axes.cend(), axis_checker);
    NODE_VALIDATION_CHECK(node,
                          invalid_axis == axes.cend(),
                          normalize_axis_error_msg(*invalid_axis, axis_checker.lower(), axis_checker.upper()));
    std::for_each(axes.begin(), axes.end(), normalize_axis_to(tensor_rank));
}

int64_t normalize_axis(const ov::Node* node, std::int64_t axis, const Rank& tensor_rank) {
    return ov::util::normalize_axis(node->description(), axis, tensor_rank);
}

int64_t normalize_axis(const std::string& node_description, std::int64_t axis, const Rank& tensor_rank) {
    if (axis < 0) {
        // Handling negative axis requires static tensor rank
        OPENVINO_ASSERT(tensor_rank.is_static(),
                        node_description,
                        " Rank must be static in order to normalize negative axis=",
                        axis);
    }
    if (tensor_rank.is_dynamic()) {
        return axis;
    }

    const auto tensor_rank_value = tensor_rank.get_length();
    return normalize_axis(node_description,
                          axis,
                          tensor_rank_value,
                          -tensor_rank_value,
                          tensor_rank_value ? (tensor_rank_value - 1) : 0);
}

int64_t normalize_axis(const ov::Node* node,
                       std::int64_t axis,
                       std::uint64_t tensor_rank,
                       std::int64_t axis_range_min,
                       std::int64_t axis_range_max) {
    return normalize_axis(node->description(), axis, tensor_rank, axis_range_min, axis_range_max);
}

int64_t normalize_axis(const std::string& node_description,
                       std::int64_t axis,
                       std::uint64_t tensor_rank,
                       std::int64_t axis_range_min,
                       std::int64_t axis_range_max) {
    // Accepted range of value for axis is [axis_range_min, axis_range_max].
    OPENVINO_ASSERT((axis_range_min <= axis) && (axis <= axis_range_max),
                    node_description,
                    normalize_axis_error_msg(axis, axis_range_min, axis_range_max));
    return normalize(axis, tensor_rank);
}
}  // namespace util
}  // namespace ov
