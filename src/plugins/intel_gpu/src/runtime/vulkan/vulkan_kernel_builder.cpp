// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vulkan_kernel_builder.hpp"

#include <cstdint>
#include <cstring>

#include "openvino/core/except.hpp"
#include "vulkan_engine.hpp"
#include "vulkan_kernel.hpp"

namespace cldnn {
namespace vulkan {

void vulkan_kernel_builder::build_kernels(const void* source,
                                          size_t source_size,
                                          KernelFormat source_format,
                                          const std::string&,
                                          std::vector<kernel::ptr>& output) const {
    OPENVINO_ASSERT(source_format == KernelFormat::NATIVE_BIN, "[GPU][Vulkan] Vulkan kernels must be provided as SPIR-V binaries");
    OPENVINO_ASSERT(source != nullptr && source_size > 0, "[GPU][Vulkan] Cannot build an empty SPIR-V kernel");

    std::vector<uint8_t> binary(source_size);
    std::memcpy(binary.data(), source, source_size);
    output.push_back(std::make_shared<vulkan_kernel>(_engine.get_vulkan_device_object(), std::move(binary), "main"));
}

}  // namespace vulkan
}  // namespace cldnn
