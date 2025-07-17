// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @file disable_fp16_comp_for_periodic_funcs.hpp
 * @brief Defines the transformation pass to disable FP16 compression for periodic functions
 */

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @ingroup ov_transformation_common_api
 * @brief DisableFP16CompressionForPeriodicFuncs is a transformation pass that disables FP16 compression
 * for specific nodes in the computation graph, particularly periodic functions like Sin and Cos.
 *
 * This transformation traverses the graph to identify nodes that modify values and disables FP16 compression
 * for those nodes. It ensures that FP16 compression is not applied to nodes where precision loss could
 * negatively impact the computation results.
 *
 * Specifically, for periodic functions like Sin and Cos, this transformation analyzes their inputs and identifies
 * the first node that performs calculations using inputs not classified as non-value-modifying nodes (as determined
 * by the is_non_value_modifying_node function). FP16 compression is disabled for such nodes to ensure computational
 * accuracy and prevent precision loss
 */
class DisableFP16CompressionForPeriodicFuncs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompressionForPeriodicFuncs");
    DisableFP16CompressionForPeriodicFuncs();
};

}  // namespace intel_gpu
}  // namespace ov
