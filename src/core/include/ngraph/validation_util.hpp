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
#include "ngraph/op/constant.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/variable_context.hpp"
#include "openvino/core/validation_util.hpp"

namespace ngraph {

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
NGRAPH_API
std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward(const Node* node,
                                                                               element::Type input_element_type,
                                                                               element::Type gamma_element_type,
                                                                               element::Type beta_element_type,
                                                                               element::Type mean_element_type,
                                                                               element::Type variance_element_type,
                                                                               const PartialShape& input_shape,
                                                                               const PartialShape& gamma_shape,
                                                                               const PartialShape& beta_shape,
                                                                               const PartialShape& mean_shape,
                                                                               const PartialShape& variance_shape);

/// \brief Apply auto padding to padding_above and padding_below inputs
///        if all needed informations are known.
///
/// \param image_shape       The shape of input image.
/// \param filter_shape      The shape of filter input.
/// \param filter_strides    The strides of applied padding.
/// \param filter_dilations  The dilations of applied padding.
/// \param pad_type          The type of padding. Auto padding is applied only
///                          for SAME_UPPER and SAME_LOWER mode.
/// \param padding_above     The beginning of padding shape.
/// \param end               The beginning of padding shape.
///
/// \return true if auto padding was applied successfully (all needed informations such as
///         spatial dims are known), false otherwise.
NGRAPH_API_DEPRECATED
NGRAPH_API
bool try_apply_auto_padding(const PartialShape& image_shape,
                            const Shape& filter_shape,
                            const Strides& filter_strides,
                            const Strides& filter_dilations,
                            const op::PadType pad_type,
                            CoordinateDiff& padding_above,
                            CoordinateDiff& padding_below);

/// \brief Try to compute the maximum value of value
/// \return (true, max_value) if can be determined, or (false, numeric_limits<uint64_t>::max())
/// if not.
/// \deprecated Use evaluate_upper_bound instead
NGRAPH_API_DEPRECATED
NGRAPH_API std::pair<bool, uint64_t> maximum_value(const Output<Node>& value);
}  // namespace ngraph
