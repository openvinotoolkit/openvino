// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/variable_context.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ngraph {
using ov::evaluate_as_partial_shape;
using ov::get_constant_from_source;
using ov::has_no_labels;
using ov::normalize_axes;
using ov::normalize_axis;
using ov::Shape;
using ov::Strides;
using ov::op::v0::Constant;

namespace element {
using ov::element::Type;
using ov::element::Type_t;
}  // namespace element

OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API
Strides conv_default_strides(const ov::Node* node,
                             const ov::PartialShape& data_batch_shape,
                             const ov::PartialShape& filters_shape);

OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API
CoordinateDiff conv_default_padding(const ov::Node* node,
                                    const ov::PartialShape& data_batch_shape,
                                    const ov::PartialShape& filters_shape);

OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API
ov::PartialShape infer_windowed_reduction_output_shape(const ov::Node* node,
                                                       const ov::PartialShape& data_shape,
                                                       const Strides& data_dilation,
                                                       const CoordinateDiff& data_padding_below,
                                                       const CoordinateDiff& data_padding_above,
                                                       const ov::PartialShape& window_shape,
                                                       const Strides& window_strides,
                                                       const Strides& window_dilation,
                                                       bool is_window_all_in_padding_allowed,
                                                       bool ceil_mode = false);

OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
void validate_conv_params_spatial_dimensions(const ov::Node* node,
                                             const size_t num_spatial_dims,
                                             const ov::op::PadType auto_pad,
                                             Strides& strides,
                                             Strides& dilations,
                                             CoordinateDiff& pads_begin,
                                             CoordinateDiff& pads_end);

OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API
ov::PartialShape infer_batched_pooling_forward(const ov::Node* node,
                                               const ov::PartialShape& data_batch_shape,
                                               const CoordinateDiff& data_padding_below,
                                               const CoordinateDiff& data_padding_above,
                                               const ov::PartialShape& window_shape,
                                               const Strides& window_strides,
                                               bool is_window_all_in_padding_allowed,
                                               bool ceil_mode = false,
                                               const Strides& window_dilation = Strides{});

OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API
ov::PartialShape infer_slice_shape(const ov::Node* node,
                                   const ov::PartialShape& input_shape,
                                   const std::vector<int64_t>& begin,
                                   const std::vector<int64_t>& end,
                                   const std::vector<int64_t>& strides,
                                   const ov::AxisSet& begin_mask,
                                   const ov::AxisSet& end_mask,
                                   const ov::AxisSet& new_axis_mask,
                                   const ov::AxisSet& shrink_axis_mask,
                                   const ov::AxisSet& ellipsis_mask);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::max()
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API std::shared_ptr<Constant> get_constant_max_of_type(element::Type_t t);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::min()
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API std::shared_ptr<Constant> get_constant_min_of_type(element::Type_t t);

/// \brief Returns a Constant storing scalar value equal to std::numeric_limits<t>::lowest()
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API std::shared_ptr<Constant> get_constant_lowest_of_type(element::Type_t t);

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
OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. "
                    "For instructions on transitioning to the new API, please refer to "
                    "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
OPENVINO_API
void infer_conv_backprop_auto_padding(const Shape& input_data_shape,
                                      const Shape& filters_shape,
                                      const Shape& output_shape,
                                      const Strides& strides,
                                      const Strides& dilations,
                                      const ov::op::PadType auto_pad_type,
                                      const CoordinateDiff& output_padding,
                                      CoordinateDiff& pads_begin,
                                      CoordinateDiff& pads_end);
}  // namespace opset1
}  // namespace ngraph

using ngraph::get_constant_from_source;
