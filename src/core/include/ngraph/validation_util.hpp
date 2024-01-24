// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <tuple>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/variable_context.hpp"

namespace ngraph {
using ov::evaluate_as_partial_shape;
using ov::get_constant_from_source;
using ov::has_no_labels;
using ov::normalize_axes;
using ov::normalize_axis;
using ov::op::v0::Constant;

namespace element {
using ov::element::Type;
using ov::element::Type_t;
}  // namespace element

NGRAPH_API_DEPRECATED
NGRAPH_API
Strides conv_default_strides(const Node* node, const PartialShape& data_batch_shape, const PartialShape& filters_shape);

NGRAPH_API_DEPRECATED
NGRAPH_API
CoordinateDiff conv_default_padding(const Node* node,
                                    const PartialShape& data_batch_shape,
                                    const PartialShape& filters_shape);

NGRAPH_API_DEPRECATED
NGRAPH_API
PartialShape infer_windowed_reduction_output_shape(const Node* node,
                                                   const PartialShape& data_shape,
                                                   const Strides& data_dilation,
                                                   const CoordinateDiff& data_padding_below,
                                                   const CoordinateDiff& data_padding_above,
                                                   const PartialShape& window_shape,
                                                   const Strides& window_strides,
                                                   const Strides& window_dilation,
                                                   bool is_window_all_in_padding_allowed,
                                                   bool ceil_mode = false);

NGRAPH_API_DEPRECATED
void validate_conv_params_spatial_dimensions(const Node* node,
                                             const size_t num_spatial_dims,
                                             const op::PadType auto_pad,
                                             Strides& strides,
                                             Strides& dilations,
                                             CoordinateDiff& pads_begin,
                                             CoordinateDiff& pads_end);

NGRAPH_API_DEPRECATED
NGRAPH_API
PartialShape infer_batched_pooling_forward(const Node* node,
                                           const PartialShape& data_batch_shape,
                                           const CoordinateDiff& data_padding_below,
                                           const CoordinateDiff& data_padding_above,
                                           const PartialShape& window_shape,
                                           const Strides& window_strides,
                                           bool is_window_all_in_padding_allowed,
                                           bool ceil_mode = false,
                                           const Strides& window_dilation = Strides{});

NGRAPH_API_DEPRECATED
NGRAPH_API
std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward(const Node* node,
                                                                               ov::element::Type input_element_type,
                                                                               ov::element::Type gamma_element_type,
                                                                               ov::element::Type beta_element_type,
                                                                               ov::element::Type mean_element_type,
                                                                               ov::element::Type variance_element_type,
                                                                               const PartialShape& input_shape,
                                                                               const PartialShape& gamma_shape,
                                                                               const PartialShape& beta_shape,
                                                                               const PartialShape& mean_shape,
                                                                               const PartialShape& variance_shape);

NGRAPH_API_DEPRECATED
NGRAPH_API
std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward(const Node* node,
                                                                               ov::element::Type input_element_type,
                                                                               ov::element::Type gamma_element_type,
                                                                               ov::element::Type beta_element_type,
                                                                               const PartialShape& input_shape,
                                                                               const PartialShape& gamma_shape,
                                                                               const PartialShape& beta_shape);

NGRAPH_API_DEPRECATED
NGRAPH_API
PartialShape infer_slice_shape(const Node* node,
                               const PartialShape& input_shape,
                               const std::vector<int64_t>& begin,
                               const std::vector<int64_t>& end,
                               const std::vector<int64_t>& strides,
                               const AxisSet& begin_mask,
                               const AxisSet& end_mask,
                               const AxisSet& new_axis_mask,
                               const AxisSet& shrink_axis_mask,
                               const AxisSet& ellipsis_mask);

/// \brief Try to compute the maximum value of value
/// \return (true, max_value) if can be determined, or (false, numeric_limits<uint64_t>::max())
/// if not.
/// \deprecated Use evaluate_upper_bound instead
NGRAPH_API_DEPRECATED
NGRAPH_API std::pair<bool, uint64_t> maximum_value(const Output<Node>& value);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::max()
NGRAPH_API_DEPRECATED
NGRAPH_API std::shared_ptr<Constant> get_constant_max_of_type(element::Type_t t);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::min()
NGRAPH_API_DEPRECATED
NGRAPH_API std::shared_ptr<Constant> get_constant_min_of_type(element::Type_t t);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::lowest()
NGRAPH_API_DEPRECATED
NGRAPH_API std::shared_ptr<Constant> get_constant_lowest_of_type(element::Type_t t);

namespace opset1 {
///
/// \brief      Calculates padding values for ConvolutionBackpropData operator.
///
/// \param[in]  input_data_shape  The input data shape.
/// \param[in]  filters_shape     The filters shape.
/// \param[in]  output_shape      The output shape defined only for spatial dimentions.
/// \param[in]  strides           The strides values.
/// \param[in]  dilations         The dilations values.
/// \param[in]  auto_pad_type     The automatic padding mode.
/// \param[in]  output_padding    The output padding values.
/// \param      pads_begin        The placeholder for paddings at the beginning of axis.
/// \param      pads_end          The placeholder for paddings at the end of axis.
///
NGRAPH_API_DEPRECATED
NGRAPH_API
void infer_conv_backprop_auto_padding(const Shape& input_data_shape,
                                      const Shape& filters_shape,
                                      const Shape& output_shape,
                                      const Strides& strides,
                                      const Strides& dilations,
                                      const op::PadType auto_pad_type,
                                      const CoordinateDiff& output_padding,
                                      CoordinateDiff& pads_begin,
                                      CoordinateDiff& pads_end);
}  // namespace opset1
}  // namespace ngraph

using ngraph::get_constant_from_source;
