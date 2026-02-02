// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API VectorizedMOE2GEMMTransposeWeights : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("VectorizedMOE2GEMMTransposeWeights");
    VectorizedMOE2GEMMTransposeWeights();
};

}  // namespace pass
}  // namespace ov
