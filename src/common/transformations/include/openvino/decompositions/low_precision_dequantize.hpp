// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/pass/node_registry.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace decomposition {

/// \brief Build the canonical low-precision dequantization sub-graph that is
///        recognised by ov::pass::MarkDequantization and downstream LPT /
///        weight-decompression passes.
///
/// The returned sub-graph implements one of:
///   y = Multiply(Convert(x, scale_type), scale)                              // symmetric
///   y = Multiply(Subtract(Convert(x, scale_type), zp), scale)                // asymmetric
/// optionally followed by a Reshape when \p output_shape is provided.
///
/// MarkDequantization matches the Multiply(Subtract(Convert(...), zp), scale)
/// / Multiply(Convert(...), scale) pattern and protects the leading Convert
/// from ConstantFolding. The optional trailing Reshape and any caller-applied
/// ConvertLike sit *outside* the matched pattern — they do not break matching,
/// but they are not themselves marked as dequantization nodes.
///
/// The output element type is taken from \p scale. If \p zero_point is given
/// and its element type differs from \p scale, a Convert is inserted on the
/// zero_point input as well — both forms (with and without that Convert) are
/// matched by ov::pass::MarkDequantization.
///
/// The helper intentionally does not append a trailing ConvertLike: that cast
/// is outside the pattern that MarkDequantization recognises and is best
/// applied by the caller when needed (e.g. mixed-precision frontends that
/// need to match the original input element type).
///
/// All nodes created by the helper are added to \p reg so the caller can
/// post-process them uniformly (e.g. PyTorch frontend iterates the registry
/// and calls NodeContext::mark_node on each entry). Callers that don't need
/// post-processing can use the overload without a NodeRegistry.
///
/// When \p output_shape is provided, a Reshape is appended only if the
/// Multiply output shape doesn't already match it (statically). This avoids
/// inserting a no-op Reshape when broadcasting already produces the desired
/// shape.
///
/// \param reg          Node registry that collects every node created by the helper.
/// \param x            Quantized input tensor (typically a low-precision Constant).
/// \param scale        Dequantization scale; its element type determines the
///                     output element type of the sub-graph.
/// \param zero_point   Optional zero point. When provided a Subtract is
///                     inserted between the Convert and the Multiply.
/// \param output_shape Optional shape constant. When provided a Reshape with
///                     special_zero=false is appended after the Multiply
///                     (skipped if the Multiply output already has that shape).
/// \param scale_decompression_precision Optional element type for the Convert node
///                     that converts the quantized scale and zp to the output element type. If not provided, the
///                     Convert node is omitted and the scale is used as-is.
ov::Output<ov::Node> TRANSFORMATIONS_API
low_precision_dequantize(ov::pass::NodeRegistry& reg,
                         const ov::Output<ov::Node>& x,
                         const ov::Output<ov::Node>& scale,
                         const ov::Output<ov::Node>& zero_point = {},
                         const ov::Output<ov::Node>& output_shape = {},
                         const ov::element::Type& decompression_precision = ov::element::dynamic);

/// \brief Convenience overload for callers that do not need access to the
///        intermediate nodes. Internally allocates a NodeRegistry and
///        forwards to the registry-based overload.
ov::Output<ov::Node> TRANSFORMATIONS_API
low_precision_dequantize(const ov::Output<ov::Node>& x,
                         const ov::Output<ov::Node>& scale,
                         const ov::Output<ov::Node>& zero_point = {},
                         const ov::Output<ov::Node>& output_shape = {},
                         const ov::element::Type& decompression_precision = ov::element::dynamic);

}  // namespace decomposition
}  // namespace ov
