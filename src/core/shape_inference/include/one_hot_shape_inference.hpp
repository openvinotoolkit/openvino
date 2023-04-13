// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/op/one_hot.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

namespace utils {
namespace one_hot {

template <class TShape>
inline bool get_data_as_shape_and_validate_sign(
    size_t idx,
    const ov::Node* op,
    TShape& shape,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        using DimType = typename TShape::value_type;
        const auto data = host_tensor_2_vector<int64_t>(constant_data.at(idx));
        shape.clear();
        std::transform(data.cbegin(), data.cend(), std::back_inserter(shape), [&](int64_t v) {
            NODE_VALIDATION_CHECK(op, v >= 0, "OneHot depth value can't be negative.");
            return static_cast<DimType>(v);
        });
        return true;
    } else {
        return get_data_as_shape<TShape>(idx, op, shape, constant_data);
    }
}

template <>
inline bool get_data_as_shape_and_validate_sign<ov::PartialShape>(
    size_t idx,
    const ov::Node* op,
    ov::PartialShape& shape,
    const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (constant_data.count(idx)) {
        const auto data = host_tensor_2_vector<int64_t>(constant_data.at(idx));
        for (const auto& value : data) {
            NODE_VALIDATION_CHECK(op, value >= 0, "OneHot depth value can't be negative.");
        }
        shape = PartialShape(data);
        return true;
    } else {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return ov::evaluate_as_partial_shape(op->input_value(idx), shape);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
}

}  // namespace one_hot
}  // namespace utils

void inline resolve_axis(OneHot* op) {
    if (op->get_input_size() < 1) {
        return;
    }
    const auto& indices_shape = op->get_input_partial_shape(0);
    if (indices_shape.rank().is_static()) {
        const auto indices_rank = indices_shape.rank().get_length();
        OPENVINO_SUPPRESS_DEPRECATED_START
        op->m_axis = ov::normalize_axis(op, op->m_axis, indices_rank + 1, -indices_rank - 1, indices_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }
}

template <class T>
void shape_infer(const OneHot* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    const auto& indices_shape = input_shapes[0];
    const auto& depth_shape = input_shapes[1];
    const auto& on_value_shape = input_shapes[2];
    const auto& off_value_shape = input_shapes[3];

    NODE_VALIDATION_CHECK(op,
                          depth_shape.is_dynamic() || ngraph::is_scalar(depth_shape.to_shape()),
                          "depth input must be scalar.");

    NODE_VALIDATION_CHECK(op,
                          on_value_shape.is_dynamic() || ngraph::is_scalar(on_value_shape.to_shape()),
                          "on_value input must be scalar.");

    NODE_VALIDATION_CHECK(op,
                          off_value_shape.is_dynamic() || ngraph::is_scalar(off_value_shape.to_shape()),
                          "off_value input must be scalar.");

    auto& result_shape = output_shapes[0];
    if (indices_shape.rank().is_static()) {
        result_shape = indices_shape;
        const auto indices_rank = indices_shape.rank().get_length();
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto axis = ov::normalize_axis(op, op->get_axis(), indices_rank + 1, -indices_rank - 1, indices_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END

        T depth_dim_as_shape;
        if (utils::one_hot::get_data_as_shape_and_validate_sign<T>(1, op, depth_dim_as_shape, constant_data) &&
            depth_dim_as_shape.size() == 1) {
            result_shape.insert(result_shape.begin() + axis, depth_dim_as_shape[0]);
        } else {
            result_shape.insert(result_shape.begin() + axis, DimType());
        }
    } else {
        result_shape = PartialShape::dynamic();
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
