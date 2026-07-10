// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <variant>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/igraph.hpp"
#include "openvino/runtime/icore.hpp"

namespace intel_npu {

class IParser {
public:
    /**
     * @brief Parses the provided binary objects and returns a wrapper over the resulted L0 handles. The model may also
     * be loaded on the device depending on the provided configuration (if "NPU_DEFER_WEIGHTS_LOAD" is off).
     *
     * @param mainBlob The core binary object to be parsed and used to build the "Graph" object.
     * @param config Used to influence the downstream flow of the implementation based on preferences.
     * @param initBlobs Optional. If provided, the "weights separation" flow is enabled and the binary objects
     * corresponding to the init schedules will be parsed as well.
     * @param model TODO
     * @return A wrapper over the corresponding L0 graph handles (multiple only if "initBlobs" has been provided). This
     * wrapper further details the compiled model and brings it in a state closer to execution.
     */
    virtual std::shared_ptr<IGraph> parse(
        const ov::Tensor& mainBlob,
        const FilteredConfig& config,
        const std::shared_ptr<ov::ICore>& core,
        std::variant<std::monostate, std::shared_ptr<const ov::Model>, std::string_view>&& weightsSource,
        const std::optional<std::vector<ov::Tensor>>& initBlobs = std::nullopt,
        const std::optional<std::string>& compatibilityDescriptor = std::nullopt) const = 0;

    virtual ~IParser() = default;
};

}  // namespace intel_npu
