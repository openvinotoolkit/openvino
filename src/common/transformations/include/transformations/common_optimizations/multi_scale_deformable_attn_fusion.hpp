// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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

}  // namespace ov::pass