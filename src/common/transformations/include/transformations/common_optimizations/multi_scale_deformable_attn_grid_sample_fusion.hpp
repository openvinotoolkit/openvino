// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {
class TRANSFORMATIONS_API MultiScaleDeformableAttnGridSampleFusion;
}  // namespace ov::pass

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses the GridSample based multi-scale deformable attention subgraph into the internal MSDA operation.
 *
 * Deformable-DETR style models (Deformable-DETR, GroundingDINO, RT-DETR) exported through the PyTorch frontend
 * sample every feature level with its own GridSample and reduce the samples with the attention weights. MSDA
 * computes the same result in one operation, without materializing the [B*H, D, Q, L, P] samples tensor.
 * Dimensions: B batch, S keys, H heads, D channels per head, Q queries, L levels, P points, h x w level size.
 *
 * Before (one chain per level l = 0 .. L-1; [ ] marks optional nodes):
 *
 *     value [B,S,H,D]                         locations [B,Q,H,L,P,2]
 *            |                                          |
 *     VariadicSplit(axis 1), output l          Multiply(2) -> Add(-1)
 *            |                                          |
 *     Reshape [B,h*w,H*D]                      Gather(l, axis 3)
 *            |                                          |
 *     Transpose(0,2,1)                         [Squeeze or Reshape]
 *            |                                          |
 *     Reshape [B*H,D,h,w]                      Transpose(0,2,1,3,4)
 *            |                                          |
 *            |                                 Reshape [B*H,Q,P,2]
 *            |                                          |
 *            +-------------- GridSample ----------------+
 *                                |
 *                  Unsqueeze/Reshape [B*H,D,Q,1,P]
 *                                |
 *                  Concat(axis -2) of the L levels         weights [B,Q,H,L,P]
 *                                |                                  |
 *                  Reshape [B*H,D,Q,L*P]                   Transpose(0,2,1,3,4)
 *                                |                                  |
 *                                |                         Reshape [B*H,1,Q,L*P]
 *                                |                                  |
 *                                +----------- Multiply -------------+
 *                                                |
 *                                   ReduceSum(axis -1) [B*H,D,Q]
 *                                                |
 *                                       Reshape [B,H*D,Q]
 *                                                |
 *                                            [Convert]
 *                                                |
 *                                    Transpose(0,2,1) [B,Q,H*D]
 *
 * The constants 2 and -1 may come through a decompression Convert. The ReduceSum may also come as
 * Multiply(AvgPool(kernel 1 x L*P, strides 1, no pads), L*P) -> [Reshape], the form ConvertReduceToPooling produces
 * (the GPU plugin runs it before this pass for f16 on devices without XMX).
 *
 * After:
 *
 *     value   spatial_shapes [L,2]   level_start_index [L]   locations   weights
 *       |          (Constant)             (Constant)              |          |
 *       +--------------+----------------------+------ MSDA -------+----------+
 *                                                      |
 *                                                  [B,Q,H*D]
 *                                                      |
 *                                     [Convert], when the original graph has one
 *
 * The rewrite is applied when:
 *  - value, locations and weights have static shapes: the matcher checks the products B*H, H*D and the sum of the
 *    level sizes on static dimension values, and the GPU MSDA primitive has a static-shape implementation only;
 *  - GridSample is bilinear with zero padding and align_corners=false, and its coordinates are 2 * x - 1 of the
 *    level-normalized locations x, which is the sampling MSDA implements (see ov_ops/msda.hpp);
 *  - level l reads output l of one VariadicSplit and locations[:, :, :, l], all levels share one value and one
 *    locations tensor of the same element type, and the level sizes add up to S;
 *  - every Transpose has the order and every Reshape the output shape shown above, so the element order matches
 *    the MSDA layout.
 */
class ov::pass::MultiScaleDeformableAttnGridSampleFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MultiScaleDeformableAttnGridSampleFusion");
    MultiScaleDeformableAttnGridSampleFusion();
};
