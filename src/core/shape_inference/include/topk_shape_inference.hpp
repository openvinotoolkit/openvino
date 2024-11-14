// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/topk.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace util {
// Helper to get correct K from tensor as shape.
template <class T>
struct GetK {
    const util::TopKBase* m_op;

    GetK(const util::TopKBase* op) : m_op{op} {}

    template <class K>
    T operator()(const K k) const {
        NODE_VALIDATION_CHECK(m_op,
                              cmp::ge(k, 0) && cmp::le(k, std::numeric_limits<T>::max()),
                              "The value of 'K' must be greater or equal to zero.",
                              " (got ",
                              k,
                              ").");
        return static_cast<T>(k);
    }
};
}  // namespace util
/**
 * \brief TopK shape inference
 *
 * \tparam TShape  Type of shape.
 *
 * \param op             Pointer to TopK operator.
 * \param input_shapes   Input shapes of TopK.
 * \param constant_data  Map of constant data. DEfault empty.
 *
 * \return Vector of output shapes for
 */
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const util::TopKBase* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    using TDim = typename TShape::value_type;
    using TDimValue = typename TDim::value_type;

    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    const auto& idx_element_type = op->get_index_element_type();
    NODE_VALIDATION_CHECK(op,
                          idx_element_type == element::i32 || idx_element_type == element::i64,
                          "Index element type attribute should be either \'i32\' or \'i64\'. Got: ",
                          idx_element_type);
    const auto& input_shape = input_shapes[0];
    const auto input_rank = input_shape.rank();

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_rank.is_dynamic() || input_rank.get_length() > 0,
                           "Input rank must be greater than 0.");

    const auto& k_shape = input_shapes[1];
    NODE_SHAPE_INFER_CHECK(op, input_shapes, k_shape.rank().compatible(0), "The 'K' input must be a scalar.");

    TRShape output_shape = input_shape;
    if (input_shape.rank().is_static()) {
        const auto normalized_axis = ov::util::try_normalize_axis(op->get_provided_axis(), input_shape.rank(), *op);
        auto& dim_axis = output_shape[normalized_axis];

        if (auto k_as_shape =
                get_input_const_data_as_shape<TRShape>(op, 1, tensor_accessor, util::GetK<TDimValue>(op))) {
            NODE_VALIDATION_CHECK(op,
                                  k_as_shape->size() == 1,
                                  "Only one value (scalar) should be provided as the 'K' input to TopK",
                                  " (got ",
                                  k_as_shape->size(),
                                  " elements).");

            const auto& k = (*k_as_shape)[0];
            if (k.is_static()) {
                dim_axis = k;
            } else {
                // in this dynamic branch we are sure of dim_axis's type
                const auto in_min = dim_axis.get_min_length();
                const auto in_max = dim_axis.get_max_length();

                const auto k_min = k.get_min_length();
                const auto k_max = k.get_max_length();

                const auto lower = std::min<TDimValue>(in_min, k_min);
                const auto upper =
                    in_max < 0
                        ? Dimension::dynamic().get_max_length()
                        : std::min<TDimValue>(in_max, (k_max < 0 ? std::numeric_limits<TDimValue>::max() : k_max));
                dim_axis = TDim(lower, upper);
            }
        } else {
            dim_axis = TDim(0, dim_axis.get_max_length());
        }
    }

    return {2, output_shape};
}
}  // namespace op
}  // namespace ov
