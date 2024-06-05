// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "graph_transformations.hpp"
#include "intel_npu/al/icompiler.hpp"

namespace intel_npu {
namespace driverCompilerAdapter {

/**
 * @brief Interface for external compiler
 * @details Isolate external API calls from general logic
 */
class IExternalCompiler {
public:
    virtual ~IExternalCompiler() = default;

    /**
     * @brief Get opset supported by compiler
     */
    virtual uint32_t getSupportedOpset() const = 0;

    /**
     * @brief Get query result for current network
     */
    virtual std::unordered_set<std::string> getQueryResult(const std::shared_ptr<const ov::Model>& model,
                                                           const Config& config) const = 0;

    /**
     * @brief Sends the serialized model and its I/O metadata to the driver for compilation.
     * @return The compiled model descriptor corresponding to the previously given network.
     */
    virtual NetworkDescription compileIR(const std::shared_ptr<const ov::Model>& model, const Config& config) const = 0;
    virtual NetworkMetadata parseBlob(const std::vector<uint8_t>& blob, const Config& config) const = 0;
};
}  // namespace driverCompilerAdapter
}  // namespace intel_npu
