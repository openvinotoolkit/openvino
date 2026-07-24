// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Adds a Clamp between an f16 Linear/FC output (MatMul with a constant weight) and a
 * residual Add, targeting FFN output projections (e.g. T5 DenseReluDense.wo) whose activations
 * can overflow f16 range and turn into NaN downstream. Clamp is expected to fuse into the
 * MatMul, so this costs almost nothing from a performance perspective.
 */
class TRANSFORMATIONS_API ClampFP16FCOutput : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ClampFP16FCOutput");
    ClampFP16FCOutput();
};

}  // namespace pass
}  // namespace ov
