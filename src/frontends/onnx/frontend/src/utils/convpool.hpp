// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace convpool {
/// \brief Get shape of kernel (filter) in pixels.
///
/// \param node The Node ptr representing Conv or Pool operation.
/// \return The kernel Shape object representing its dimensions (height, width, depth).
ov::Shape get_kernel_shape(const ov::frontend::onnx::Node& node);

///
/// \brief      Get number of pixels to stride operation by in each direction.
///
/// \param[in]  node         The Node ptr representing Conv or Pool operation.
/// \param[in]  kernel_rank  The operator's kernel rank.
///
/// \return     The kernel Shape object representing its dimensions (height, width,
///             depth).
ov::Strides get_strides(const ov::frontend::onnx::Node& node, const std::size_t kernel_rank = 0UL);

///
/// \brief      Get number of pixels for filter dilation in each direction.
///
/// \param[in]  node         The Node ptr representing ONNX operation.
/// \param[in]  kernel_rank  The operator'skernel rank.
///
/// \return     The ov::Strides object containing number of pixels for filter dilation
///             (height, width, depth).
ov::Strides get_dilations(const ov::frontend::onnx::Node& node, const std::size_t kernel_rank = 0UL);

/// \brief      Gets the 'ceil_mode' (rounding type) attribute value.
///
/// \param[in]  node  The ONNX node we query for attribute.
///
/// \return     The OV RoundingType object representing 'ceil_mode' attribute value.
ov::op::RoundingType get_rounding_type(const ov::frontend::onnx::Node& node);

/// \brief Get padding values for the operation described by an ONNX node.
/// \details Values are taken from the `pads` attribute.
///
///          `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...].
///
/// \param node The Node ptr representing ONNX operation.
/// \param kernel_rank The rank of the kernel which we retrieve pads for.
///
/// \return A pair of (padding_above, padding_below), which elements contains number of
///         pixels to pad in respective dimensions (height, width, depth).
std::pair<ov::CoordinateDiff, ov::CoordinateDiff> get_pads(const ov::frontend::onnx::Node& node,
                                                           const size_t kernel_rank);

/// \brief Get padding values for the operation described by an ONNX node.
/// \details Values are taken from the `pads` attribute.
///
///          `pads` value should follow [x1_begin, x2_begin..., x1_end, x2_end,...].
///
/// \param node The Node ptr representing ONNX operation.
///
/// \return A pair of (padding_above, padding_below), which elements contains number of
///         pixels to pad in respective dimensions (height, width, depth).
std::pair<ov::CoordinateDiff, ov::CoordinateDiff> get_pads(const ov::frontend::onnx::Node& node);

///
/// \brief         Calculate paddings with respect to auto_pad value.
///
/// \param[in]     data_shape     The input data tensor shape.
/// \param[in]     filter_shape   The input filters tensor shape.
/// \param[in]     strides        The data strides.
/// \param[in]     dilations      The data dilations.
/// \param[in]     pad_type       The value of auto_pad attribute.
/// \param[in,out] padding_below  The paddings below axis.
/// \param[in,out] padding_above  The paddings above axis.
///
/// \see        ov::op::PadType
void calculate_auto_pads(const ov::Shape& data_shape,
                         const ov::Shape& filter_shape,
                         const ov::Strides& strides,
                         const ov::Strides& dilations,
                         const ov::op::PadType& pad_type,
                         ov::CoordinateDiff& padding_below,
                         ov::CoordinateDiff& padding_above);

/// \brief      Gets the 'auto_pad' attribute value.
///
/// \param[in]  node  The ONNX node we query for attribute.
///
/// \return     The OV PadType object representing 'auto_pad' attribute value.
///
ov::op::PadType get_auto_pad(const ov::frontend::onnx::Node& node);

/// \brief      Reshape group convolution filters to match desired shape:
///             from [C_INPUT x C_OUTPUT/groups x k1 x k2 x ... x kn]
///             to [GROUPS, C_INPUT, C_OUTPUT, K_D, ..., K_1]
///
/// \param[in]  filters     Filter input to reshape
/// \param[in]  groups      Number of groups
///
/// \return     Reshaped filters input.
ov::Output<ov::Node> get_reshaped_filters(const ov::Output<ov::Node>& filters, int64_t groups);
}  // namespace convpool
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
