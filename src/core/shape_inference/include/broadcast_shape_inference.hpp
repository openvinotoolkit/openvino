// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/axis_vector.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/util/broadcast_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template <typename T, class TRShape = result_shape_t<T>>
void validate_target_shape_none(const ov::Node* op,
                                const T& arg_shape,
                                const AxisVector& axes_mapping_val,
                                const TRShape& target_input_shape) {
    if (arg_shape.rank().is_static() && target_input_shape.rank().is_static()) {
        const auto target_rank_length = target_input_shape.size();
        // axes_mapping needs to be in sorted order
        NODE_VALIDATION_CHECK(op,
                              std::is_sorted(axes_mapping_val.begin(), axes_mapping_val.end()),
                              "Broadcast doesn't permit transposes. axes_mapping ",
                              axes_mapping_val,
                              " not in sorted order");

        if (arg_shape.size() == 0 && axes_mapping_val.size() > 0) {
            NODE_VALIDATION_CHECK(op,
                                  target_input_shape[axes_mapping_val[0]].compatible(1),
                                  "Broadcast target[axes_mapping[0]]. Expected 1. Got ",
                                  target_input_shape[axes_mapping_val[0]]);
        }

        for (size_t i = 0; i < axes_mapping_val.size(); ++i) {
            NODE_VALIDATION_CHECK(op,
                                  axes_mapping_val[i] < target_rank_length,
                                  "Broadcast axes_mapping[",
                                  i,
                                  "]: ",
                                  axes_mapping_val[i],
                                  " exceeds target rank ",
                                  target_rank_length);

            if (arg_shape.size() > 0) {
                NODE_VALIDATION_CHECK(
                    op,
                    target_input_shape[axes_mapping_val[i]].compatible(arg_shape[i]) || arg_shape[i].compatible(1),
                    "Broadcast target[axes_mapping[",
                    i,
                    "]]",
                    " Expected ",
                    arg_shape[i],
                    ". Got ",
                    target_input_shape[axes_mapping_val[i]]);
            }
        }
    }
}

template <typename T, class TRShape = result_shape_t<T>>
void validate_target_shape_numpy(const ov::Node* op, const T& arg_shape, const TRShape& target_input_shape) {
    if (arg_shape.rank().is_dynamic() || target_input_shape.rank().is_dynamic()) {
        return;
    }
    const auto arg_rank_length = arg_shape.size();
    const auto target_rank_length = target_input_shape.size();
    NODE_VALIDATION_CHECK(op,
                          !(target_rank_length < arg_rank_length),
                          "Broadcast target_shape has smaller rank ",
                          target_rank_length,
                          " than arg shape ",
                          arg_rank_length);
    const size_t start_axis = target_rank_length - arg_rank_length;
    for (auto i = start_axis; i < target_rank_length; ++i) {
        NODE_VALIDATION_CHECK(op,
                              arg_shape[i - start_axis].is_dynamic() || target_input_shape[i].is_dynamic() ||
                                  arg_shape[i - start_axis].compatible(1) ||
                                  arg_shape[i - start_axis].compatible(target_input_shape[i]),
                              "Input shape dimension equal ",
                              arg_shape[i - start_axis],
                              " cannot be broadcasted (numpy mode) to ",
                              target_input_shape[i],
                              ". Allowed input dimension value would be 1",
                              target_input_shape[i] != 1 ? " or " : "",
                              target_input_shape[i] != 1 ? std::to_string(target_input_shape[i].get_length()) : "");
    }
}

template <typename T, class TRShape = result_shape_t<T>>
void set_result_shape_pdpd(const ov::Node* op,
                           const T& arg0_shape,
                           const TRShape& target_input_shape,
                           TRShape& result_shape,
                           const ov::op::BroadcastModeSpec& broadcast_spec) {
    using DimType = typename T::value_type;
    if (arg0_shape.rank().is_dynamic() || target_input_shape.rank().is_dynamic()) {
        result_shape = PartialShape::dynamic(target_input_shape.rank());
        return;
    }
    result_shape = target_input_shape;
    auto start_axis = ((broadcast_spec.m_type == op::BroadcastType::PDPD) && (broadcast_spec.m_axis == -1))
                          ? static_cast<int64_t>(target_input_shape.size()) - static_cast<int64_t>(arg0_shape.size())
                          : broadcast_spec.m_axis;

    NODE_VALIDATION_CHECK(op, start_axis >= 0, "Broadcast start_axis must be greater than 0");

    for (size_t i = start_axis; i < target_input_shape.size(); ++i) {
        const auto& arg_dim = arg0_shape[i - start_axis];
        if (arg_dim == 1) {
            result_shape[i] = target_input_shape[i];
        } else if (target_input_shape[i] == 1) {
            result_shape[i] = arg_dim;
        } else {
            NODE_VALIDATION_CHECK(op,
                                  DimType::merge(result_shape[i], arg_dim, target_input_shape[i]),
                                  "Broadcast incorrect target shape. Expecting either 1 or ",
                                  arg_dim,
                                  " . Got ",
                                  target_input_shape[i]);
        }
    }
}

template <typename T, class TRShape = result_shape_t<T>>
void set_result_shape_bidirectional(const ov::Node* op,
                                    const T& arg_shape,
                                    TRShape& target_input_shape,
                                    TRShape& result_shape) {
    using DimType = typename T::value_type;
    if (arg_shape.rank().is_dynamic() || target_input_shape.rank().is_dynamic()) {
        result_shape = PartialShape::dynamic();
        return;
    }
    auto arg_shape_vec = static_cast<TRShape>(arg_shape);

    // Add left padding to the shape with smaller rank, if the ranks are not equal
    if (arg_shape_vec.size() < target_input_shape.size()) {
        arg_shape_vec.insert(arg_shape_vec.begin(), target_input_shape.size() - arg_shape_vec.size(), 1);
    } else {
        target_input_shape.insert(target_input_shape.begin(), arg_shape_vec.size() - target_input_shape.size(), 1);
    }

    result_shape.resize(target_input_shape.size());

    for (size_t i = 0; i < target_input_shape.size(); ++i) {
        NODE_VALIDATION_CHECK(op,
                              DimType::broadcast_merge(result_shape[i], arg_shape_vec[i], target_input_shape[i]),
                              "Broadcast incorrect target shape. Expecting either 1 or ",
                              arg_shape_vec[i],
                              ". Got ",
                              target_input_shape[i]);
    }
}

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> broadcast_base_shape_infer(const ov::op::util::BroadcastBase* op,
                                                const std::vector<T>& input_shapes,
                                                const ITensorAccessor& ta = make_tensor_accessor()) {
    // shape node should produce a one dimensional shape.
    auto broadcast_shape_rank = input_shapes[1].rank();
    NODE_VALIDATION_CHECK(op,
                          broadcast_shape_rank.compatible(1),
                          "Broadcast shape rank must be 1, but has ",
                          broadcast_shape_rank);

    const auto& mode = op->get_broadcast_spec();
    if (mode.m_type == BroadcastType::NONE) {
        // axes_mapping node should produce a one dimensional shape.
        auto axes_shape_rank = input_shapes[2].rank();
        NODE_VALIDATION_CHECK(op,
                              axes_shape_rank.compatible(1),
                              "Broadcast axes rank must be 1, but has ",
                              axes_shape_rank);
    }

    auto output_shapes = std::vector<TRShape>(1);
    auto& result_shape = output_shapes[0];
    const auto& data_input_shape = input_shapes[0];
    const auto& target_input_shape = input_shapes[1];
    const bool is_target_input_shape_static = target_input_shape.is_static();

    auto target_as_shape = get_input_const_data_as_shape<TRShape>(op, 1, ta);

    if (!target_as_shape) {
        if (auto concat = ov::as_type_ptr<ov::opset1::Concat>(op->get_input_node_shared_ptr(1))) {
            const auto concat_inputs = concat->input_values();
            if (concat->get_output_partial_shape(0).is_static() && concat->get_shape().size() == 1 &&
                concat_inputs.size() == shape_size(concat->get_shape())) {
                target_as_shape.emplace();
                for (const auto& concat_input : concat_inputs) {
                    auto source_node_ptr = concat_input.get_node_shared_ptr();
                    if (auto source_const_ptr = ov::as_type_ptr<ov::opset1::Constant>(source_node_ptr)) {
                        target_as_shape->push_back(source_const_ptr->get_axis_vector_val()[0]);
                    } else {
                        target_as_shape->push_back(Dimension::dynamic());
                    }
                }
            }
        }
    }

    if (mode.m_type == BroadcastType::NONE) {
        if (target_as_shape) {
            result_shape = *target_as_shape;
        } else if (is_target_input_shape_static) {
            result_shape = PartialShape::dynamic(target_input_shape[0].get_length());
        } else {
            result_shape = PartialShape::dynamic();
        }
        // Validate axes_mapping
        const auto& axes_shape = input_shapes[2];
        if (data_input_shape.rank().is_static() && target_input_shape.rank().is_static() && axes_shape.is_static()) {
            int64_t input_rank =
                (data_input_shape.size() == 0 && axes_shape[0].get_length() > 0) ? 1 : data_input_shape.size();
            NODE_VALIDATION_CHECK(
                op,
                static_cast<uint64_t>(axes_shape[0].get_length()) == static_cast<uint64_t>(input_rank),
                "Broadcast axes_mapping shape ",
                axes_shape,
                " doesn't match rank of input tensor ",
                input_rank);

            if (target_as_shape) {
                if (auto axes_mapping = get_input_const_data_as<T, size_t, AxisVector>(op, 2, ta)) {
                    validate_target_shape_none(op, data_input_shape, *axes_mapping, *target_as_shape);
                }
            }
        }
    } else if (mode.m_type == BroadcastType::NUMPY) {
        if (target_as_shape) {
            result_shape = *target_as_shape;
            validate_target_shape_numpy(op, data_input_shape, *target_as_shape);
        } else if (is_target_input_shape_static) {
            result_shape = PartialShape::dynamic(target_input_shape[0].get_length());
        } else {
            result_shape = PartialShape::dynamic();
        }
    } else if (mode.m_type == BroadcastType::PDPD) {
        if (target_as_shape) {
            set_result_shape_pdpd(op, data_input_shape, *target_as_shape, result_shape, mode);
        } else if (is_target_input_shape_static) {
            result_shape = PartialShape::dynamic(target_input_shape[0].get_length());
        } else {
            result_shape = PartialShape::dynamic();
        }
    } else if (mode.m_type == BroadcastType::BIDIRECTIONAL) {
        if (target_as_shape) {
            set_result_shape_bidirectional(op, data_input_shape, *target_as_shape, result_shape);
        } else if (data_input_shape.rank().is_static() && is_target_input_shape_static) {
            result_shape = PartialShape::dynamic(target_input_shape[0].get_length());
            // The logic of BroadcastType::BIDIRECTIONAL matches broadcast_merge_into with AutoBroadcastType::NUMPY
            NODE_VALIDATION_CHECK(
                op,
                TRShape::broadcast_merge_into(result_shape, data_input_shape, op::AutoBroadcastType::NUMPY),
                "Broadcast shape inference failed, output shape calculation with "
                "'broadcast_merge_into' was unsuccessful.");
        } else {
            result_shape = PartialShape::dynamic();
        }
    }
    return output_shapes;
}
}  // namespace util

namespace v3 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const ov::op::v3::Broadcast* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    auto& mode = op->get_broadcast_spec();
    if (mode.m_type == BroadcastType::NONE) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes.size() == 3,
                              "axes_mapping input should be provided if explicit mode is used");
    } else {
        NODE_VALIDATION_CHECK(op,
                              input_shapes.size() == 2,
                              "axes_mapping input should not be provided for mode other than explicit");
    }
    return broadcast_base_shape_infer(op, input_shapes, ta);
}
}  // namespace v3

namespace v1 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const ov::op::v1::Broadcast* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3));

    return broadcast_base_shape_infer(op, input_shapes, ta);
}
}  // namespace v1

}  // namespace op
}  // namespace ov
