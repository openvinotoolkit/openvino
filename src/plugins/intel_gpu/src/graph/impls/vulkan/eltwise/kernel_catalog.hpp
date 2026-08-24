// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "intel_gpu/runtime/kernel_args.hpp"
#include "kernel_kind.hpp"

namespace cldnn::vulkan::eltwise_detail {

std::vector<std::shared_ptr<kernel_string>> make_kernel_sources(kernel_kind kind, uint32_t fused_chain_length);

}  // namespace cldnn::vulkan::eltwise_detail
