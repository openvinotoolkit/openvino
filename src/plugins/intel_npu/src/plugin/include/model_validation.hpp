// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"

namespace intel_npu {

/**
 * @brief Rejects models the NPU compiler cannot handle: those with at least one parameter or output dimension that is
 * dynamic and has no finite upper bound (upper bound == INT64_MAX).
 *
 * Such fully unbounded dimensions are common in LLM exports (e.g. via optimum-cli). Without this guard they reach the
 * VPUX compiler and surface as opaque errors (signed-integer overflow in broadcast analysis, "to_shape was called on a
 * dynamic shape"). This validation throws an actionable ov::Exception that directs the user to reshape the model.
 *
 * Bounded dynamic dimensions (finite upper bound) are intentionally allowed through, as they may work with
 * DYNAMIC_SHAPE_TO_STATIC or future compiler support.
 *
 * @param model The model to validate.
 * @throws ov::Exception if any parameter or output dimension is dynamic with no finite upper bound.
 */
void validate_no_unbounded_dynamic_dimensions(const std::shared_ptr<const ov::Model>& model);

}  // namespace intel_npu
