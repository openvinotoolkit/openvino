// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/multi_matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

class TRANSFORMATIONS_API MultiScaleDeformableAttnFusion;

}  // namespace ov::pass

namespace ov::pass {

class MultiScaleDeformableAttnFusion : public ov::pass::MultiMatcher {
public:
    OPENVINO_RTTI("MultiScaleDeformableAttnFusion");

    MultiScaleDeformableAttnFusion();
};

// Exact layout emitted by the GroundingDINO OpenVINO export (VariadicSplit + dynamic
// Reshape shape inputs + projection MatMul with transpose_a), not the older PR graph.
class TRANSFORMATIONS_API GroundingDinoMSDAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GroundingDinoMSDAFusion");

    GroundingDinoMSDAFusion();
};

}  // namespace ov::pass