// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/pass/node_registry.hpp"

namespace ov {
namespace decompositions {

/// \brief Build a reference RMSNorm decomposition matched by ov::pass::RMSFusion.
///
/// The returned sub-graph implements:
///   y = x * Power(Sqrt(ReduceMean(x^2, axes) + eps), -1)            // when scale is empty
///   y = scale * (x * Power(Sqrt(ReduceMean(x^2, axes) + eps), -1))  // when scale is provided
///
/// The shape of the resulting graph is intentionally aligned with the pattern
/// detected by ov::pass::RMSFusion so that this decomposition is always fused
/// back into ov::op::internal::RMS during plugin compilation. This is the
/// canonical building block to be used by frontends (e.g. PyTorch aten::rms_norm)
/// instead of emitting a hand-rolled sub-graph.
///
/// All nodes created by the helper are added to \p reg so the caller can
/// post-process them uniformly (e.g. PyTorch frontend iterates the registry
/// and calls NodeContext::mark_node on each entry).
///
/// \param reg    Node registry that collects every node created by the helper.
/// \param x      Normalized input tensor.
/// \param axes   Reduction axes for ReduceMean. RMSFusion currently fuses only
///               the last-dimension reduction (axes constant with a single
///               element equal to -1 or rank-1).
/// \param eps    Scalar epsilon constant added before the square root. Should
///               share element type with \p x.
/// \param scale  Optional scaling tensor (gamma). Pass an empty Output to skip
///               the trailing multiplication.
ov::Output<ov::Node> rms_norm(ov::pass::NodeRegistry& reg,
                              const ov::Output<ov::Node>& x,
                              const ov::Output<ov::Node>& axes,
                              const ov::Output<ov::Node>& eps,
                              const ov::Output<ov::Node>& scale = {});

}  // namespace decompositions
}  // namespace ov
