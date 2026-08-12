// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel_builder.hpp"

#include "openvino/core/except.hpp"

namespace cldnn {
namespace vulkan {

void vulkan_kernel_builder::build_kernels(const void*, size_t, KernelFormat, const std::string&, std::vector<kernel::ptr>&) const {
    OPENVINO_THROW("[GPU][Vulkan] SPIR-V kernel construction is not implemented yet");
}

}  // namespace vulkan
}  // namespace cldnn
