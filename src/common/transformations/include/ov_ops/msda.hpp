// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {
/// @brief Multi-scale deformable attention of Deformable-DETR style detectors.
///
/// Inputs:
///   value               [B, S, H, D] float: the keys of all feature levels, level after level;
///   value_spatial_shapes [L, 2] integer: (h, w) of every level, with S = sum(h * w);
///   level_start_index   [L] integer: index of the first key of every level in S;
///   sampling_locations  [B, Q, H, L, P, 2] float: (x, y) of every sampling point in level-normalized
///                       coordinates: [0, 1] spans the level, points outside it read zeros;
///   attention_weights   [B, Q, H, L, P] float.
/// value, sampling_locations and attention_weights have one element type.
///
/// Output [B, Q, H * D]: out[b, q, h * D + d] = sum over l, p of
///   attention_weights[b, q, h, l, p] * sample_l(x * w_l - 0.5, y * h_l - 0.5)[b, h, d],
/// where sample_l is the bilinear interpolation of the level l keys of value with zeros outside the level, that is
/// GridSample with align_corners=false and zero padding at the coordinates 2 * (x, y) - 1.
class TRANSFORMATIONS_API MSDA : public ov::op::Op {
public:
    OPENVINO_OP("MSDA", "ie_internal_opset", ov::op::Op);

    MSDA() = default;

    MSDA(const Output<Node>& value,
         const Output<Node>& value_spatial_shapes,
         const Output<Node>& level_start_index,
         const Output<Node>& sampling_locations,
         const Output<Node>& attention_weights);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
