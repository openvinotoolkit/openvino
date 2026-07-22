// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"
#include "../sycl_device.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cldnn {
namespace sycl {
namespace intel {

class sycl_kernel_builder : public kernel_builder {
public:
    explicit sycl_kernel_builder(const sycl_device& device);

    void build_kernels(const void* src,
                       size_t src_bytes,
                       KernelFormat src_format,
                       const std::string& options,
                       std::vector<kernel::ptr>& out) const override;

    const sycl_device& _device;
};

}  // namespace intel
}  // namespace sycl
}  // namespace cldnn
