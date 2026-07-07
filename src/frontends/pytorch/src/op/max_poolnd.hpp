// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/node_registry.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

// Build the static-kernel max pool (v14::MaxPool) for an already-resolved kernel/strides/pads/
// dilations. Shared by the translator (constant kernel) and the deferred resolver (kernel that
// became constant after shape propagation). Nodes go into `rg`; returns the pool result (2 outputs
// when `return_indices`).
OutputVector build_static_max_pool(ov::pass::NodeRegistry& rg,
                                   Output<Node> input,
                                   int dims,
                                   bool return_indices,
                                   const ov::Shape& kernel,
                                   const ov::Strides& strides,
                                   const ov::Shape& pads,
                                   const ov::Strides& dilations,
                                   ov::op::RoundingType rounding_type);

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
