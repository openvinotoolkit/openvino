// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @brief Force FP32 precision for specific layer types (debug feature)
 *
 * This transformation inserts FP16->FP32 converts before specified layer types
 * and FP32->FP16 converts after them, forcing those layers to compute in FP32
 * even when the model is configured for FP16 inference.
 *
 * Usage:
 *   export GPU_FORCE_FP32_LAYER_TYPES="MatMul RMSNorm"
 *   export GPU_FORCE_FP32_LAYER_NAMES="block.19 ff.net.0.proj"
 *
 * Matching rules:
 *   types empty + names empty  => no-op
 *   types empty + names set    => name pattern match only
 *   types set   + names empty  => type match only (original behavior)
 *   types set   + names set    => AND (intersection) of both
 *
 * This is a debug feature for isolating FP16 accuracy issues.
 */
class ForceFP32Selective : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ForceFP32Selective");

    ForceFP32Selective(const std::vector<std::string>& forced_types,
                       const std::vector<std::string>& forced_names = {})
        : m_forced_types(forced_types), m_forced_names(forced_names) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::vector<std::string> m_forced_types;
    std::vector<std::string> m_forced_names;
};

}  // namespace intel_gpu
}  // namespace ov
