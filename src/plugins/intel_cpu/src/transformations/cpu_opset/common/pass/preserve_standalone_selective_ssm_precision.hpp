// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

/*
 * Description:
 *     PreserveStandaloneSelectiveSSMPrecision keeps standalone SelectiveSSM and PagedSelectiveSSM operations in the
 *     precision provided by the model. It marks the operations and their inputs to prevent ConvertPrecision from
 *     changing data, state, metadata, or output element types. The pass detects Paged Attention models and leaves them
 *     unchanged, preserving their configured inference and cache precision.
 *
 * Before:
 *
 *     data inputs [0..5]       metadata inputs [6..10]
 *              \                         /
 *               +-----------------------+
 *               |     SelectiveSSM      |  or  PagedSelectiveSSM
 *               +-----------------------+
 *                           |
 *                         output
 *
 * After:
 *
 *     data inputs [0..5]       metadata inputs [6..10]
 *       [no conversion]          [no conversion]
 *              \                         /
 *               +-----------------------+
 *               |     SelectiveSSM      |  or  PagedSelectiveSSM
 *               |    [no conversion]    |
 *               +-----------------------+
 *                           |
 *                 original-precision output
 *
 *     SelectiveSSM uses only data inputs [0..5].
 */
class PreserveStandaloneSelectiveSSMPrecision final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PreserveStandaloneSelectiveSSMPrecision");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

/*
 * Description:
 *     RestoreStandalonePagedSelectiveSSMStatePrecision runs after ConvertPagedAttnInputs. That transformation may set
 *     the mutable state parameter to the configured inference precision. For a standalone operation, the core
 *     PagedSelectiveSSM contract requires the state and data inputs to have the same element type. This pass restores
 *     the state parameter to the preserved data precision before model validation.
 *
 * Before:
 *
 *     data inputs [0..4]: T       state parameter: inference precision
 *                  \                         /
 *                   +-----------------------+
 *                   |   PagedSelectiveSSM   |   state precision may differ from T
 *                   +-----------------------+
 *
 * After:
 *
 *     data inputs [0..4]: T             state parameter: T
 *                  \                         /
 *                   +-----------------------+
 *                   |   PagedSelectiveSSM   |
 *                   +-----------------------+
 *                               |
 *                           output: T
 */
class RestoreStandalonePagedSelectiveSSMStatePrecision final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RestoreStandalonePagedSelectiveSSMStatePrecision");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
