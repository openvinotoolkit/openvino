// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "dimension_util.hpp"
#include "fft_common_validation.hpp"
#include "openvino/core/axis_vector.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/op/util/fft_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace fft {
template <class TRShape, typename std::enable_if<std::is_same<TRShape, PartialShape>::value>::type* = nullptr>
void apply_dims_from_sizes(const util::FFTBase* op,
                           TRShape& output_shape,
                           const std::vector<int64_t>& axes,
                           const ITensorAccessor& ta) {
    using namespace ov::util;
    using DimType = typename TRShape::value_type;

    if (const auto output_bounds = get_input_bounds<TRShape, int64_t>(op, 2, ta)) {
        const auto minus_one_bound = std::make_pair(dim::inf_bound, dim::inf_bound);
        const auto num_of_axes = axes.size();
        const auto& symbols =
            op->get_input_size() > 2 ? op->get_input_source_output(2).get_tensor().get_value_symbol() : TensorSymbol();
        const bool propagate_symbols = num_of_axes <= symbols.size();
        for (size_t i = 0; i < num_of_axes; ++i) {
            if ((*output_bounds)[i] != minus_one_bound) {
                auto& out_dim = output_shape[(axes)[i]];
                out_dim = DimType((*output_bounds)[i].first, (*output_bounds)[i].second);
                if (propagate_symbols && symbols[i] != nullptr) {
                    out_dim.set_symbol(symbols[i]);
                }
            }
        }
    } else {
        for (auto axis : axes) {
            output_shape[axis] = ov::Dimension::dynamic();
        }
    }
}

template <class TRShape, typename std::enable_if<!std::is_same<TRShape, PartialShape>::value>::type* = nullptr>
void apply_dims_from_sizes(const util::FFTBase* op,
                           TRShape& output_shape,
                           const std::vector<int64_t>& axes,
                           const ITensorAccessor& ta) {
    using namespace ov::util;
    using DimType = typename TRShape::value_type;

    if (const auto output_dim_vals = get_input_const_data_as<TRShape, int64_t>(op, 2, ta)) {
        const auto num_of_axes = axes.size();
        for (size_t i = 0; i < num_of_axes; ++i) {
            if ((*output_dim_vals)[i] != dim::inf_bound) {
                output_shape[(axes)[i]] = DimType((*output_dim_vals)[i]);
            }
        }
    }
}
}  // namespace fft

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const util::FFTBase* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3));

    const auto& input_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    auto axes = get_input_const_data_as<TRShape, int64_t>(op, 1, ta);

    util::fft_common_validation::shape_validation(op,
                                                  input_shapes,
                                                  axes,
                                                  util::fft_common_validation::FFTKind::ComplexInput);

    output_shape = input_shape;
    if (input_shapes.size() == 3 && input_shape.rank().is_static()) {
        if (axes) {
            ov::op::fft::apply_dims_from_sizes(op, output_shape, *axes, ta);
        } else {
            for (size_t i = 0; i < input_shape.size() - 1; ++i) {
                output_shape[i] = ov::Dimension::dynamic();
            }
        }
    }
    return output_shapes;
}
}  // namespace op
}  // namespace ov
