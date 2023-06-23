// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_convolution.hpp"

#include <cmath>
#include <cstddef>
#include <ngraph/validation_util.hpp>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/runtime/host_tensor.hpp"

namespace ov {
namespace intel_gna {
namespace op {
namespace internal {

int64_t calculate_num_spatial(const GNAConvolution* op,
                              const ov::PartialShape& input_shape,
                              const ov::PartialShape& filters_shape,
                              const int64_t& num_non_spatial_data_dims,
                              const int64_t& num_non_spatial_filter_dims) {
    int64_t num_spatial = op->m_num_spatial;
    if (num_spatial == -1) {
        const auto& input_rank = input_shape.rank();
        const auto& filters_rank = filters_shape.rank();

        if (const auto& size = op->m_dilations.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto& size = op->m_strides.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto& size = op->m_pads_begin.size())
            num_spatial = static_cast<int64_t>(size);
        if (const auto& size = op->m_pads_end.size())
            num_spatial = static_cast<int64_t>(size);
        if (input_rank.is_static())
            num_spatial = input_rank.get_length() - num_non_spatial_data_dims;
        if (filters_rank.is_static())
            num_spatial = filters_rank.get_length() - num_non_spatial_filter_dims;
    }
    return num_spatial;
}

void update_and_validate_attributes(GNAConvolution* op) {
    const auto& num_spatial = op->m_num_spatial;
    if (num_spatial != -1) {
        auto& strides = op->m_strides;
        auto& dilations = op->m_dilations;
        auto& pad_begin = op->m_pads_begin;
        auto& pad_end = op->m_pads_end;
        auto& auto_pad = op->m_auto_pad;

        if (strides.empty())
            strides = ov::Strides(num_spatial, 1);
        if (dilations.empty())
            dilations = ov::Strides(num_spatial, 1);
        if (pad_begin.empty() || auto_pad == ov::op::PadType::VALID)
            pad_begin = ov::CoordinateDiff(num_spatial, 0);
        if (pad_end.empty() || auto_pad == ov::op::PadType::VALID)
            pad_end = ov::CoordinateDiff(num_spatial, 0);

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
                                          [](const size_t& i) {
                                              return i > 0;
                                          }),
                              "Filter dilation (",
                              dilations,
                              ") has zero dimension.");
        NODE_VALIDATION_CHECK(op,
                              std::all_of(strides.begin(),
                                          strides.end(),
                                          [](const size_t& i) {
                                              return i > 0;
                                          }),
                              "Filter strides (",
                              strides,
                              ") has zero dimension.");
    }
}

// code is based on ngraph/core/shape_inference/include/convolution_shape_inference.hpp
// but instead of NCHW uses NHWC layout

template <class T>
inline bool dynamic_check(const int64_t& num_spatial) {
    OPENVINO_ASSERT(num_spatial != -1,
                    "Convolution shape inference doesn't have enough information for static shape calculation");
    return true;
}

// FIXME: do we need that function as a template ?
template <>
inline bool dynamic_check<ov::PartialShape>(const int64_t& num_spatial) {
    return num_spatial != -1;
}

// FIXME: do we need that function as a template ?
// TODO: search where that function is used in openvino
template <class T>
bool resolve_auto_pad_for_shape(const GNAConvolution* op,
                                ov::CoordinateDiff& pads_begin,
                                ov::CoordinateDiff& pads_end,
                                const std::vector<T>& input_shapes,
                                const int64_t& num_non_spatial_data_dims,
                                const int64_t& num_non_spatial_filter_dims) {
    const auto& auto_pad = op->get_auto_pad();
    if (auto_pad != ov::op::PadType::SAME_UPPER && auto_pad != ov::op::PadType::SAME_LOWER) {
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
        const auto& input_dim = input_shape[i + 1];
        const auto& filters_dim = filters_shape[i + 1];
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

            pads_begin[i] = auto_pad == ov::op::PadType::SAME_UPPER ? padding_lhs : padding_rhs;
            pads_end[i] = auto_pad == ov::op::PadType::SAME_UPPER ? padding_rhs : padding_lhs;
        } else {
            status = false;
        }
    }
    return status;
}

// FIXME: do we need that function as a template ?
// TODO: search where that function is used in openvino
template <class T>
void shape_infer(const GNAConvolution* op,
                 const ov::CoordinateDiff& pads_begin,
                 const ov::CoordinateDiff& pads_end,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);
    auto input_shape = input_shapes[0], filters_shape = input_shapes[1];

    const auto& num_spatial = op->m_num_spatial;
    NODE_VALIDATION_CHECK(op,
                          num_spatial != -1,
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
    // Channel is the last in NHWC layout
    *(output_shape.rbegin()) = filters_shape[0];  // NHWC C is last instead of filters_shape[0] for NCHW layout

    const auto n_data_channel = *(input_shape.rbegin());
    const auto n_filter_channel = *(filters_shape.rbegin());

    NODE_VALIDATION_CHECK(
        op,
        n_data_channel.compatible(n_filter_channel),  // instead of input_shape[1].compatible(filters_shape[1]),
        "Data batch channel count (",
        n_data_channel,  // instead of input_shape[1],
        ") does not match filter input ",
        "channel count (",
        n_filter_channel,  // instead of filters_shape[1],
        ").");

    const auto& dilations = op->m_dilations;
    const auto& strides = op->m_strides;

    for (int64_t i = 0; i < num_spatial; ++i) {
        const auto& input_dim = input_shape[i + 1];
        const auto& filters_dim = filters_shape[i + 1];
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
            output_shape[i + 1] = (data_padded_dilated_dim - window_dilated_dim) / strides[i] + 1;
        }
    }
}

}  // namespace internal

GNAConvolution::GNAConvolution(const ov::Output<Node>& data_batch,
                               const ov::Output<Node>& filters,
                               const ov::Output<Node>& bias,
                               const ov::Strides& strides,
                               const ov::CoordinateDiff& pads_begin,
                               const ov::CoordinateDiff& pads_end,
                               const ov::Strides& dilations,
                               const ov::op::PadType& auto_pad)
    : ov::op::Op({data_batch, filters, bias}),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad) {
    constructor_validate_and_infer_types();
}

GNAConvolution::GNAConvolution(const ov::Output<Node>& data_batch,
                               const ov::Output<Node>& filters,
                               const ov::Strides& strides,
                               const ov::CoordinateDiff& pads_begin,
                               const ov::CoordinateDiff& pads_end,
                               const ov::Strides& dilations,
                               const ov::op::PadType& auto_pad)
    : ov::op::Op({data_batch, filters}),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad) {
    constructor_validate_and_infer_types();
}

bool GNAConvolution::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

void GNAConvolution::validate_and_infer_types() {
    ov::element::Type data_batch_et = get_input_element_type(0);
    ov::element::Type filters_et = get_input_element_type(1);

    ov::element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          ov::element::Type::merge(result_et, data_batch_et, filters_et),
                          "Element types for data batch and filters do not match (data batch element type: ",
                          data_batch_et,
                          ", filters element type: ",
                          filters_et,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element types must be numeric. Got: ",
                          result_et);
    auto& data_shape = get_input_partial_shape(0);
    auto& filter_shape = get_input_partial_shape(1);

    m_num_spatial = internal::calculate_num_spatial(this, data_shape, filter_shape, 2, 2);
    internal::update_and_validate_attributes(this);

    std::vector<ov::PartialShape> input_shapes = {data_shape, filter_shape};
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};

    if (m_num_spatial != -1) {
        internal::resolve_auto_pad_for_shape(this, m_pads_begin, m_pads_end, input_shapes, 2, 2);
        internal::shape_infer(this, m_pads_begin, m_pads_end, input_shapes, output_shapes);
    }

    set_output_type(0, result_et, output_shapes[0]);
}

std::shared_ptr<ov::Node> GNAConvolution::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    if (new_args.size() == 2) {
        return std::make_shared<GNAConvolution>(new_args.at(0),
                                                new_args.at(1),
                                                m_strides,
                                                m_pads_begin,
                                                m_pads_end,
                                                m_dilations,
                                                m_auto_pad);
    } else if (new_args.size() == 3) {
        return std::make_shared<GNAConvolution>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                m_strides,
                                                m_pads_begin,
                                                m_pads_end,
                                                m_dilations,
                                                m_auto_pad);
    }

    OPENVINO_THROW("Unsupported number of arguments for GNAConvolution operation");
}
}  // namespace op
}  // namespace intel_gna
}  // namespace ov