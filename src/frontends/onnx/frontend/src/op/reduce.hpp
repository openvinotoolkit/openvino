// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_13 {
/// \brief      Compute the sum of the input tensor's elements along the provided
///             axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor has the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_sum(const ov::frontend::onnx::Node& node);
}  // namespace set_13
namespace set_1 {
/// \brief      Compute the log sum of the input tensor's elements along the
///             provided axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_log_sum(const ov::frontend::onnx::Node& node);

/// \brief      Compute the log sum exponent of the input tensor's elements along
///             the provided axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_log_sum_exp(const ov::frontend::onnx::Node& node);

/// \brief      Compute the L1 norm of the input tensor's element along the provided
///             axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_l1(const ov::frontend::onnx::Node& node);

/// \brief      Compute the L2 norm of the input tensor's element along the provided
///             axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_l2(const ov::frontend::onnx::Node& node);

/// \brief      Compute the maximum value of the input tensor's elements along the
///             provided axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_max(const ov::frontend::onnx::Node& node);

/// \brief      Compute the mean value of the input tensor's elements along the
///             provided axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_mean(const ov::frontend::onnx::Node& node);

/// \brief      Compute the minimum value of the input tensor's elements along the
///             provided axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_min(const ov::frontend::onnx::Node& node);

/// \brief      Compute the product of the input tensor's elements along the
///             provided axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_prod(const ov::frontend::onnx::Node& node);

/// \brief      Compute the sum of the input tensor's elements along the provided
///             axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_sum(const ov::frontend::onnx::Node& node);

/// \brief      Compute the sum square of the input tensor's element along the
///             provided axes.
///
/// \par Overview
///     The output tensor has the same rank as the input if Node attribute keepdims
///     equals 1. If keepdims equals 0, then the output tensor have the reduced
///     dimension pruned.
///
/// \param[in]  node  The ONNX node representing operation.
///
/// \return     The OV node equivalent of the ONNX operation.
///
ov::OutputVector reduce_sum_square(const ov::frontend::onnx::Node& node);

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
