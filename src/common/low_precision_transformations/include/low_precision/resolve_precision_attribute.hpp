// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "low_precision/layer_transformation.hpp"
#include "low_precision/lpt_visibility.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief ResolvePrecisionAttribute resolves multi-precision PrecisionsAttribute
 * on FakeQuantize outputs to a single precision based on the FQ's output ranges.
 *
 * After MarkupOptimizations propagates a set of possible PrecisionsAttribute values
 * (e.g. {u8, i8}), this pass visits each FakeQuantize and selects the most suitable
 * precision from the existing set, based on the FQ’s output_low / output_high ranges
 * (signed vs. unsigned).
 * The selection prefers representations that avoid unnecessary zero-points, resulting
 * in a single, resolved precision for downstream consumers.
 */
class LP_TRANSFORMATIONS_API ResolvePrecisionAttribute : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("low_precision::ResolvePrecisionAttribute");
    ResolvePrecisionAttribute();

    static void filterPrecisionsAttribute(std::shared_ptr<ov::op::v0::FakeQuantize> layer);
    static DataPrecision getDataPrecision(std::shared_ptr<ov::op::v0::FakeQuantize> layer);
};

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
