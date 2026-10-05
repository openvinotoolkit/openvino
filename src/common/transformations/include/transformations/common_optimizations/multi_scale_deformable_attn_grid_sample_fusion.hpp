// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// Fuses the GridSample based formulation of multi-scale deformable attention
// into the internal MSDA operation. The pattern is produced by Deformable-DETR
// style detectors and segmenters (Deformable-DETR, GroundingDINO, RT-DETR,
// Mask2Former) exported through the PyTorch frontend. The number of feature
// levels, sampling points, heads and embedding channels is derived from the
// graph shapes, so no architecture specific constant is matched.
class TRANSFORMATIONS_API MultiScaleDeformableAttnGridSampleFusion : public MatcherPass {
public:
    OPENVINO_RTTI("MultiScaleDeformableAttnGridSampleFusion");
    MultiScaleDeformableAttnGridSampleFusion();
};

}  // namespace ov::pass
