// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Preserves i64 values consumed by PagedSelectiveSSM metadata inputs.
 *
 * CPU converts general i64 computations to i32. PagedSelectiveSSM accepts i64 metadata and may consume values that
 * cannot be represented by i32. This pass keeps only the i64 producer paths that reach the operation's metadata
 * inputs. When such a producer is shared, its other consumers retain the normal CPU policy through an explicit i32
 * conversion at the boundary of the protected path.
 *
 * Before:
 *
 *     i64 producer path ----+----> PagedSelectiveSSM metadata
 *                           |
 *                           +----> unrelated consumer
 *
 * After:
 *
 *     i64 producer path ---------> PagedSelectiveSSM metadata
 *                    |
 *                    +-- Convert(i32) --> unrelated consumer
 */
class PreservePagedSelectiveSSMMetadataPrecision final : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PreservePagedSelectiveSSMMetadataPrecision");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::intel_cpu
