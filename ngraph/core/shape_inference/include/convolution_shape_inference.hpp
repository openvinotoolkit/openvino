// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <openvino/op/convolution.hpp>

namespace ov {
namespace op {
namespace v1 {

int64_t calculate_num_spatial(const Convolution* op,
                              const PartialShape& input_shape,
                              const PartialShape& filters_shape,
                              const int64_t& num_non_spatial_data_dims,
                              const int64_t& num_non_spatial_filter_dims);
void update_and_validate_attributes(Convolution* op);

template <class T>
inline bool dynamic_check(const int64_t& num_spatial) {
    OPENVINO_ASSERT(num_spatial != -1,
                    "Convolution shape inference doesn't have enough information for static shape calculation");
    return true;
}

template<>
inline bool dynamic_check<PartialShape>(const int64_t& num_spatial) {
    return num_spatial != -1;
}

template<class T>
bool resolve_auto_pad_for_shape(const Convolution* op,
                                CoordinateDiff& pads_begin,
                                CoordinateDiff& pads_end,
                                const std::vector<T> &input_shapes,
                                const int64_t& num_non_spatial_data_dims,
                                const int64_t& num_non_spatial_filter_dims) {
    const auto& auto_pad = op->get_auto_pad();
    if (auto_pad != op::PadType::SAME_UPPER && auto_pad != op::PadType::SAME_LOWER) {
        pads_begin = op->m_pads_begin;
        pads_end = op->m_pads_end;
        return true;
    }

    auto& num_spatial = op->m_num_spatial;
    if (!dynamic_check<T>(num_spatial))
        return false;

    auto input_shape = input_shapes[0];
    auto filters_shape = input_shapes[1];

    if (input_shape.rank().is_dynamic())
        input_shape.resize(num_spatial + num_non_spatial_data_dims);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(num_spatial + num_non_spatial_filter_dims);

    const auto& strides = op->m_strides;
    const auto& dilations = op->m_dilations;
    pads_begin.resize(num_spatial);
    pads_end.resize(num_spatial);

    bool status = true;
    for (int64_t i = 0; i < num_spatial; ++i) {
        const auto& input_dim = input_shape[i + 2];
        const auto& filters_dim = filters_shape[i + 2];
        if (input_dim.is_static() && filters_dim.is_static()) {
            const int64_t& window_dilated_dim = (filters_dim.get_length() - 1) * dilations[i] + 1;
            NODE_VALIDATION_CHECK(op,
                                  window_dilated_dim > 0,
                                  "Window after dilation has dimension less than 1 (dim: ",
                                  window_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");

            const int64_t& image_size = input_dim.get_length();
            const int64_t& filter_stride = strides[i];
            const int64_t& output_size = (image_size + filter_stride - 1) / filter_stride;

            const int64_t& tmp = (output_size - 1) * filter_stride + window_dilated_dim;
            const int64_t& padding_needed = tmp > image_size ? tmp - image_size : 0;

            const size_t& padding_lhs = static_cast<size_t>(padding_needed / 2);
            const size_t& padding_rhs = static_cast<size_t>(padding_needed - padding_lhs);

            pads_begin[i] = auto_pad == op::PadType::SAME_UPPER ? padding_lhs : padding_rhs;
            pads_end[i] = auto_pad == op::PadType::SAME_UPPER ? padding_rhs : padding_lhs;
        } else {
            status = false;
        }
    }
    return status;
}


template<class T>
void shape_infer(const Convolution* op,
                 const CoordinateDiff& pads_begin,
                 const CoordinateDiff& pads_end,
                 const std::vector<T> &input_shapes,
                 std::vector<T> &output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    auto input_shape = input_shapes[0], filters_shape = input_shapes[1];

    const auto& num_spatial = op->m_num_spatial;
    NODE_VALIDATION_CHECK(op, num_spatial != -1,
                          "Convolution shape_infer should be provided with correct num_spatial attribute");

    if (input_shape.rank().is_dynamic())
        input_shape.resize(num_spatial + 2);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(num_spatial + 2);

    NODE_VALIDATION_CHECK(op,
                          (static_cast<int64_t>(input_shape.size()) == (num_spatial + 2)) &&
                          (static_cast<int64_t>(filters_shape.size()) == (num_spatial + 2)),
                          "Data batch and filters rank do not match (data batch shape: ",
                          input_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");

    // ranks are originally static or aligned with num_spatial, attributes assumed to be valid
    auto& output_shape = output_shapes[0];
    output_shape.resize(num_spatial + 2);
    output_shape[0] = input_shape[0];
    output_shape[1] = filters_shape[0];

    NODE_VALIDATION_CHECK(
            op,
            input_shape[1].compatible(filters_shape[1]),
            "Data batch channel count (",
            input_shape[1],
            ") does not match filter input ",
            "channel count (",
            filters_shape[1],
            ").");

    const auto& dilations = op->m_dilations;
    const auto& strides = op->m_strides;

    for (int64_t i = 0; i < num_spatial; ++i) {
        const auto& input_dim = input_shape[i + 2];
        const auto& filters_dim = filters_shape[i + 2];
        if (input_dim.is_static() && filters_dim.is_static()) {
            const int64_t& window_dilated_dim = (filters_dim.get_length() - 1) * dilations[i] + 1;
            NODE_VALIDATION_CHECK(op,
                                  window_dilated_dim > 0,
                                  "Window after dilation has dimension less than 1 (dim: ",
                                  window_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");

            const int64_t& data_padded_dilated_dim = input_dim.get_length() + pads_begin[i] + pads_end[i];
            NODE_VALIDATION_CHECK(op,
                                  window_dilated_dim <= data_padded_dilated_dim,
                                  "Window after dilation has dimension (dim: ",
                                  window_dilated_dim,
                                  ") larger than the data shape after padding (dim: ",
                                  data_padded_dilated_dim,
                                  ") at axis ",
                                  i,
                                  ".");
            output_shape[i + 2] = (data_padded_dilated_dim - window_dilated_dim) / strides[i] + 1;
        }
    }
}

}
}
}
