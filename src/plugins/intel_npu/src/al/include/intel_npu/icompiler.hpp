// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include "intel_npu/config/config.hpp"
#include "intel_npu/network_metadata.hpp"
#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu {

#ifndef ICOMPILER_MAKE_VERSION
/// @brief Generates npu compiler (generic 'oneAPI') API version number
#    define ICOMPILER_MAKE_VERSION(_major, _minor) ((_major << 16) | (_minor & 0x0000ffff))
#endif  // ICOMPILER_MAKE_VERSION

/**
 * @struct NetworkDescription
 * @brief The object returned by the compiler
 * to provide such information about a network as description of inputs and outputs,
 * name and compiled network in a format executable by device
 */
struct NetworkDescription final {
    NetworkDescription(std::vector<uint8_t>&& compiledNetwork, NetworkMetadata&& metadata)
        : compiledNetwork(std::move(compiledNetwork)),
          metadata(std::move(metadata)) {}
    NetworkDescription(ov::Tensor&& compiledNetWorkTensor, NetworkMetadata&& metadata)
        : compiledNetwork(),
          metadata(std::move(metadata)),
          compiledNetworkTensor(std::move(compiledNetWorkTensor)) {}
    // Force move semantics to prevent blob copies
    NetworkDescription(const NetworkDescription&) = delete;
    NetworkDescription(NetworkDescription&&) = default;
    NetworkDescription& operator=(const NetworkDescription&) = delete;
    NetworkDescription& operator=(NetworkDescription&&) = default;
    ~NetworkDescription() = default;

    std::vector<uint8_t> compiledNetwork;

    NetworkMetadata metadata;

    ov::Tensor compiledNetworkTensor;
};

}  // namespace intel_npu
