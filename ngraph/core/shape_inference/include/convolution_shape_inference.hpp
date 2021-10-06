// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/convolution.hpp>


namespace ov {
namespace op {
namespace v1 {


template<class T>
void shape_infer(Convolution* op, const std::vector<T> &input_shapes, std::vector<T> &output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    auto input_shape = input_shapes[0], filters_shape = input_shapes[1];

    auto& dilations = op->m_dilations;
    auto& strides = op->m_strides;
    auto& num_spatial = op->m_num_spatial;
    auto& pad_begin = op->m_pads_begin, &pad_end = op->m_pads_end;
    const auto& auto_pad = op->m_auto_pad;

    calculate_num_spatial_dims_and_update_attributes(op, input_shape, filters_shape, dilations,
                                                     strides, pad_begin, pad_end, auto_pad, num_spatial);

    if (num_spatial < 1)
        return;
    // ranks are originally static or aligned with num_spatials, attributes are valid
    auto& output_shape = output_shapes[0];
    output_shape.resize(num_spatial + 2);
    output_shape[0] = input_shape[0];
    output_shape[1] = filters_shape[0];

    NODE_VALIDATION_CHECK(
            op,
            input_shape[1].is_dynamic() || filters_shape[1].is_dynamic() || input_shape[1] == filters_shape[1],
            "Data batch channel count (",
            input_shape[1],
            ") does not match filter input ",
            "channel count (",
            filters_shape[1],
            ").");

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
            if (auto_pad == op::PadType::SAME_UPPER || auto_pad == op::PadType::SAME_LOWER) {
                const int64_t& image_size = input_dim.get_length();
                const int64_t& filter_stride = strides[i];
                const int64_t& output_size = (image_size + filter_stride - 1) / filter_stride;

                const int64_t& tmp = (output_size - 1) * filter_stride + window_dilated_dim;
                const int64_t& padding_needed = tmp > image_size ? tmp - image_size : 0;

                const size_t& padding_lhs = static_cast<size_t>(padding_needed / 2);
                const size_t& padding_rhs = static_cast<size_t>(padding_needed - padding_lhs);

                pad_begin[i] = auto_pad == op::PadType::SAME_UPPER ? padding_lhs : padding_rhs;
                pad_end[i] = auto_pad == op::PadType::SAME_UPPER ? padding_rhs : padding_lhs;
            }

            const int64_t& data_padded_dilated_dim = input_dim.get_length() + pad_begin[i] + pad_end[i];
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

template <class ShapeType>
void calculate_num_spatial_dims_and_update_attributes(Convolution* op,
                                                      ShapeType& input_shape,
                                                      ShapeType& filters_shape,
                                                      Strides& dilations,
                                                      Strides& strides,
                                                      CoordinateDiff& pad_begin,
                                                      CoordinateDiff& pad_end,
                                                      const op::PadType& auto_pad,
                                                      int64_t& num_spatial) {
    const auto &input_rank = input_shape.rank();
    const auto &filters_rank = filters_shape.rank();
    if (num_spatial == -1) {
        if (const auto &size = dilations.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto &size = strides.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto &size = pad_begin.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto &size = pad_end.size())
            num_spatial = static_cast<int64_t>(size);
        if (input_rank.is_static())
            num_spatial = input_rank.get_length() - 2;
        if (filters_rank.is_static())
            num_spatial = filters_rank.get_length() - 2;

        if (num_spatial == -1)
            return;  // can not deduce output rank

        if (strides.empty())
            strides = Strides(num_spatial, 1);
        if (dilations.empty())
            dilations = Strides(num_spatial, 1);
        if (pad_begin.empty() || auto_pad == op::PadType::VALID)
            pad_begin = CoordinateDiff(num_spatial, 0);
        if (pad_end.empty() || auto_pad == op::PadType::VALID)
            pad_end = CoordinateDiff(num_spatial, 0);

        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(strides.size()) == num_spatial,
                              "Strides should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(dilations.size()) == num_spatial,
                              "Dilations should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(pad_begin.size()) == num_spatial &&
                              static_cast<int64_t>(pad_end.size()) == num_spatial,
                              "Pads should be defined for all and only spatial features.");
        NODE_VALIDATION_CHECK(op,
                              std::all_of(dilations.begin(),
                                          dilations.end(),
                                          [](const size_t &i) {
                                              return i > 0;
                                          }),
                              "Filter dilation (",
                              dilations,
                              ") has zero dimension.");
        NODE_VALIDATION_CHECK(op,
                              std::all_of(strides.begin(),
                                          strides.end(),
                                          [](const size_t &i) {
                                              return i > 0;
                                          }),
                              "Filter strides (",
                              strides,
                              ") has zero dimension.");
    }

    if (input_rank.is_dynamic())
        input_shape.resize(num_spatial + 2);
    if (filters_rank.is_dynamic())
        filters_shape.resize(num_spatial + 2);

    NODE_VALIDATION_CHECK(op,
                          (static_cast<int64_t>(input_shape.size()) == (num_spatial + 2)) &&
                          (static_cast<int64_t>(filters_shape.size()) == (num_spatial + 2)),
                          "Data batch and filters rank do not match (data batch shape: ",
                          input_shape,
                          ", filters shape: ",
                          filters_shape,
                          ").");
}

}
}
}
