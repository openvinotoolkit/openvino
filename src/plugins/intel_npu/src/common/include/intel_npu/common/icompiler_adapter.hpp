// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/igraph.hpp"

namespace intel_npu {

class ICompilerAdapter {
public:
    virtual std::shared_ptr<IGraph> compile(const std::shared_ptr<const ov::Model>& model,
                                            const Config& config) const = 0;

    /**
     * @brief TODO
     *
     * @param model
     * @param config
     * @return std::shared_ptr<IGraph>
     */
    virtual std::shared_ptr<IGraph> compileWS(const std::shared_ptr<ov::Model>& model, const Config& config) const = 0;

    /**
     * @brief Parses the provided binary objects and returns a wrapper over the resulted L0 handles. The model may also
     * be loaded on the device depending on the provided configuration (if "NPU_DEFER_WEIGHTS_LOAD" is off).
     *
     * @param mainBlob The core binary object to be parsed and used to build the "Graph" object.
     * @param blobAllocatedByPlugin Information required for managing the memory region allocated for the compiled
     * model.
     * @param config Used to influence the downstream flow of the implementation based on preferences.
     * @param initBlobs Optional. If provided, the "weights separation" flow is enabled and the binary objects
     * corresponding to the init schedules will be parsed as well.
     * @param model Optional, but required if "initBlobs" is provided. The "ov::Model" object is leveraged in the
     * "weights separation" implementation in order to extract the buffers of the weights.
     * @return A wrapper over the corresponding L0 graph handles (multiple only if "initBlobs" has been provided). This
     * wrapper further details the compiled model and brings it in a state closer to execution.
     */
    virtual std::shared_ptr<IGraph> parse(
        ov::Tensor mainBlob,
        const bool blobAllocatedByPlugin,
        const Config& config,
        std::optional<std::vector<ov::Tensor>> initBlobs = std::nullopt,
        const std::optional<std::shared_ptr<const ov::Model>>& model = std::nullopt) const = 0;

    virtual ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;
    virtual uint32_t get_version() const = 0;
    virtual std::vector<std::string> get_supported_options() const = 0;
    virtual bool is_option_supported(std::string optname) const = 0;

    virtual ~ICompilerAdapter() = default;
};

}  // namespace intel_npu
