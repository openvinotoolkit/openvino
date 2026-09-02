// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "host_compile_mode.hpp"

#include <algorithm>

#include "intel_npu/config/options.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

bool enable_host_compile_if_needed(const std::shared_ptr<const ov::Model>& model,
                                   FilteredConfig& config,
                                   const Logger& logger) {
    // Automatic selection applies only to Plugin compiler requests without an explicit compilation mode or
    // dynamic-to-static conversion.
    if (model == nullptr || config.get<COMPILER_TYPE>() != ov::intel_npu::CompilerType::PLUGIN ||
        config.has<COMPILATION_MODE>() || config.get<DYNAMIC_SHAPE_TO_STATIC>()) {
        return false;
    }

    // HostCompile allocates dynamic buffers from I/O upper bounds, so every dynamic dimension must be bounded.
    const auto hasFiniteUpperBounds = [](const auto& port) {
        const auto& shape = port.get_partial_shape();
        const auto rank = shape.rank();
        return rank.is_static() && std::all_of(shape.begin(), shape.end(), [](const ov::Dimension& dimension) {
                   return dimension.get_interval().has_upper_bound();
               });
    };

    // Detect a bounded dynamic 4D I/O port that makes the model a HostCompile candidate.
    const auto isDynamicHostCompilePort = [&hasFiniteUpperBounds](const auto& port) {
        const auto& shape = port.get_partial_shape();
        const auto rank = shape.rank();

        if (!(shape.is_dynamic() && rank.is_static() && rank.get_length() == 4 && hasFiniteUpperBounds(port))) {
            return false;
        }

        // Assumed N,C,H,W order. Only height (H) and width (W) determine candidacy - channel (C) is not considered,
        // and batch (N) alone does not matter either. Accepted patterns: H, W, HW, NH, NW and NHW; a dynamic batch
        // (N) with both H and W static ("N alone") is the only combination that is rejected.
        return shape[2].is_dynamic() || shape[3].is_dynamic();
    };

    const auto& modelInputs = model->inputs();
    const auto& modelOutputs = model->outputs();
    const bool inputsDynamic = std::any_of(modelInputs.begin(), modelInputs.end(), isDynamicHostCompilePort);
    const bool outputsDynamic = std::any_of(modelOutputs.begin(), modelOutputs.end(), isDynamicHostCompilePort);

    // Candidate detection above uses any_of; validate every I/O separately because one unrelated unbounded port
    // still prevents HostCompile from allocating all dynamic buffers.
    const bool allPortsHaveFiniteUpperBounds =
        std::all_of(modelInputs.begin(), modelInputs.end(), hasFiniteUpperBounds) &&
        std::all_of(modelOutputs.begin(), modelOutputs.end(), hasFiniteUpperBounds);

    if (inputsDynamic && outputsDynamic && allPortsHaveFiniteUpperBounds) {
        logger.info("NPU_COMPILATION_MODE not set; selecting 'HostCompile_Interpreter' "
                    "for fully-dynamic model (inputs and outputs both dynamic)");
        config.update({{ov::intel_npu::compilation_mode.name(), "HostCompile_Interpreter"}});
        return true;
    }

    return false;
}

bool uses_host_compile_dynamic_graph(const std::shared_ptr<const ov::Model>& model, const Config& config) {
    return model != nullptr && model->is_dynamic() &&
           config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::PLUGIN &&
           config.get<COMPILATION_MODE>().find("HostCompile") == 0;
}

}  // namespace intel_npu
