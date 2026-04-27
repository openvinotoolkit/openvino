// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/pass/node_registry.hpp"

namespace ov {
namespace decompositions {

/// \brief Build the canonical (simplest) RoPE decomposition that is
///        guaranteed to be fused by ov::pass::RoPEFusion into
///        ov::op::internal::RoPE.
///
/// The returned sub-graph implements (along the last axis):
///   first_half, second_half = split(x, 2)
///   first_  = first_half  * cos - second_half * sin
///   second_ = second_half * cos + first_half  * sin
///   y       = concat(first_, second_, axis=-1)
///
/// The negation is expressed as Multiply(-1) + Add (not Subtract) because that
/// is the exact pattern the RoPEFusion matcher accepts.
///
/// All nodes created by the helper are added to \p reg so the caller can
/// post-process them uniformly (e.g. PyTorch frontend iterates the registry
/// and calls NodeContext::mark_node on each entry).
///
/// Frontends (e.g. ONNX com.microsoft.RotaryEmbedding) are expected to call
/// this helper for the core formula and add their own pre/post-processing
/// (position_ids gather, interleaved-mode shuffle, 3D<->4D layout conversion).
///
/// \param reg             Node registry that collects every node created by the helper.
/// \param x               4-D input tensor of shape [bs, num_heads, seqlen, head_size].
/// \param cos             Cosine cache broadcastable to [?, 1, ?, head_size/2].
/// \param sin             Sine cache broadcastable to [?, 1, ?, head_size/2].
/// \param half_head_size  Number of element pairs per head, i.e. head_size / 2.
ov::Output<ov::Node> rope(ov::pass::NodeRegistry& reg,
                          const ov::Output<ov::Node>& x,
                          const ov::Output<ov::Node>& cos,
                          const ov::Output<ov::Node>& sin,
                          int64_t half_head_size);

}  // namespace decompositions
}  // namespace ov
