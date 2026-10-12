// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/pass/pass.hpp"
#ifdef GPU_DEBUG_CONFIG

namespace ov {
namespace intel_gpu {


/**
 * @brief Promote selected FP16 graph inputs and outputs to FP32 for accuracy debugging.
 *
 * Requires ENABLE_DEBUG_CAPS. Runs after GPU precision conversion and fusion;
 * filters use the current operation type and friendly name at this stage.
 *
 * Environment examples (leave the other filter unset for single-filter use):
 *   OV_GPU_FORCE_FP32_LAYER_TYPES="FullyConnected RMS" ./benchmark_app -m model.xml -d GPU
 *   OV_GPU_FORCE_FP32_LAYER_NAMES="block.19 ff.net.0.proj" ./benchmark_app -m model.xml -d GPU
 *
 * Values are space-separated. Types match exactly; names match substrings.
 * Matching is case-sensitive, with OR within each list and AND between lists.
 * Empty filters or no matching FP16 outputs produce no changes.
 * Parameter, Result, Placeholder, and outputless nodes are not initial targets.
 *
 * Connected pass-through operations with FP16 outputs are repeatedly added
 * to the region, including branches outside the filters. Region-internal edges stay FP32;
 * external FP16 inputs are converted and region exits restore FP16.
 * Only originally FP16 outputs are promoted; other output types are preserved.
 * Recognized FP16 precision attributes are updated on supported operations.
 *
 * Constant data, state, and subgraph boundaries are not fully supported;
 * subgraph bodies and separate calculation precision attributes are not updated.
 * Widening FP16 values cannot recover precision already lost. Changed-output
 * logs (OV_VERBOSE=2) do not establish the eventual GPU kernel precision.
 * See src/plugins/intel_gpu/docs/gpu_debug_utils.md for scope and examples.
 */
class ForceFP32Selective : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ForceFP32Selective");

    ForceFP32Selective(const std::vector<std::string>& forced_types,
                       const std::vector<std::string>& forced_names)
        : m_forced_types(forced_types), m_forced_names(forced_names) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::vector<std::string> m_forced_types;
    std::vector<std::string> m_forced_names;
};


}  // namespace intel_gpu
}  // namespace ov

#endif // GPU_DEBUG_CONFIG
