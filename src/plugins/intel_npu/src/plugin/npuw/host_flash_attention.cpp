// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "host_flash_attention.hpp"

#include "logging.hpp"
#include "openvino/openvino.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {
namespace function {

std::optional<HostFlashAttention> HostFlashAttention::from(const std::shared_ptr<ov::Model>& model) {
    LOG_INFO("Attempting to create HostFlashAttention from model");
    LOG_BLOCK();

    // TODO: Implement the logic to extract Host Flash Attention information from the model
    // For now, return nullopt as placeholder
    LOG_WARN("HostFlashAttention::from is not yet implemented");

    return std::nullopt;
}

}  // namespace function

namespace compiled {

// Constructor implementation - extracts metadata
HostFlashAttention::HostFlashAttention(const function::HostFlashAttention& func_hfa) {
    LOG_INFO("Constructing compiled::HostFlashAttention");
    LOG_BLOCK();

    // TODO: Extract metadata from function::HostFlashAttention
    LOG_WARN("compiled::HostFlashAttention constructor is not yet implemented");
}

}  // namespace compiled

}  // namespace npuw
}  // namespace ov
