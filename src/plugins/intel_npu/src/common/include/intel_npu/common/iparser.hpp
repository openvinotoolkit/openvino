// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"

namespace intel_npu {

class IParser {
public:
    /**
     * @brief Parses the provided binary objects and returns a wrapper over the resulted L0 handles. The model may also
     * be loaded on the device depending on the provided configuration (if "NPU_DEFER_WEIGHTS_LOAD" is off).
     *
     * @param zeroInitStruct The level zero structures.
     * @param mainBlob The core binary object to be parsed and used to build the "Graph" object.
     * @param config Used to influence the downstream flow of the implementation based on preferences.
     * @param initBlobs Optional. If provided, the "weights separation" flow is enabled and the binary objects
     * corresponding to the init schedules will be parsed as well.
     * @param model Optional, but required if "initBlobs" is provided. The "ov::Model" object is leveraged in the
     * "weights separation" implementation in order to extract the buffers of the weights.
     * @return A wrapper over the corresponding L0 graph handles (multiple only if "initBlobs" has been provided). This
     * wrapper further details the compiled model and brings it in a state closer to execution.
     */
    virtual std::shared_ptr<IGraph> parse(
        const ov::Tensor& mainBlob,
        const FilteredConfig& config,
        const std::optional<std::vector<ov::Tensor>>& initBlobs = std::nullopt,
        std::optional<std::shared_ptr<const ov::Model>>&& model = std::nullopt) const = 0;

    virtual ~IParser() = default;
};

}  // namespace intel_npu
