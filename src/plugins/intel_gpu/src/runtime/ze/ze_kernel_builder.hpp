// Copyright (C) 2016-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"
#include "intel_gpu/runtime/device.hpp"

#include "ze_device.hpp"
#include "ze_kernel.hpp"
#include "ze_common.hpp"

#include <mutex>
namespace cldnn {
namespace ocl {
struct ocl_kernel_builder;
struct ocl_device;
}  // namespace ocl
namespace ze {

class ze_kernel_builder : public kernel_builder{
public:
    ze_kernel_builder(const ze_device &device) : m_device(device) {}
    void build_kernels(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options, std::vector<kernel::ptr> &out) const override;

private:
    /// @brief Check if L0 can build kernels from source
    bool check_l0_build_support() const;
    /// @brief Build module through L0 API
    std::shared_ptr<ze_module_holder> build_module_l0(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options) const;
    /// @brief Build module through OCL API and repackage to L0 module
    std::shared_ptr<ze_module_holder> build_module_ocl(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options) const;
    void init_ocl_builder() const;
    const ze_device &m_device;
    // OCL workaround for legacy devices that does not support l0 compilation
    mutable std::shared_ptr<::cldnn::ocl::ocl_kernel_builder> m_ocl_builder;
    mutable std::shared_ptr<::cldnn::ocl::ocl_device> m_ocl_device;
    mutable std::mutex m_mutex;

};
}  // namespace ze
}  // namespace cldnn

