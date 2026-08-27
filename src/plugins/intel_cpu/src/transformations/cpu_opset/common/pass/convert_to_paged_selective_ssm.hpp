// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Allows PagedSelectiveSSM to use an independent recurrent-state storage precision.
 *
 * ConvertPagedAttnInputs selects the state-table storage precision independently from the computation precision.
 * This pass preserves that contract during Core validation while keeping the operation type and output precision.
 *
 *     A, dt, B, x, C: T                  A, dt, B, x, C: T
 *     state: T_STATE          ->          state: T_STATE
 *       PagedSelectiveSSM                   TypeRelaxed<PagedSelectiveSSM>
 *       output: T                            output: T
 */
class ConvertToPagedSelectiveSSM final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertToPagedSelectiveSSM");

    ConvertToPagedSelectiveSSM();
};

}  // namespace ov::intel_cpu
