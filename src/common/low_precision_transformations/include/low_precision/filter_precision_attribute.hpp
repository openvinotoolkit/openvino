// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "low_precision/lpt_visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief FilterPrecisionAttribute resolves multi-precision PrecisionsAttribute
 * on FakeQuantize outputs to a single precision based on the FQ's output ranges.
 *
 * After MarkupOptimizations propagates PrecisionsAttribute (e.g., {u8, i8}) across
 * the graph, this pass visits each FakeQuantize and narrows the attribute to a single
 * precision determined by the FQ's output_low/output_high values (signed vs unsigned).
 * This ensures that downstream consumers (e.g., Convolution) see a resolved single
 * precision in the shared attribute before FakeQuantizeDecomposition runs.
 */
class LP_TRANSFORMATIONS_API FilterPrecisionAttribute : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("low_precision::FilterPrecisionAttribute");
    FilterPrecisionAttribute();
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
