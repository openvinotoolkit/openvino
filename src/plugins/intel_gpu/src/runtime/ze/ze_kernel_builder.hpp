// Copyright (C) 2018-2026 Intel Corporation
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
class ocl_kernel_builder;
struct ocl_device;
}  // namespace ocl
namespace ze {

class ze_kernel_builder : public kernel_builder{
public:
    ze_kernel_builder(const ze_device &device) : m_device(device) {}
    void build_kernels(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options, std::vector<kernel::ptr> &out) const override;

private:
    /// @brief Check if ZE can build kernels from source
    bool check_ze_build_support() const;
    /// @brief Build kernels through ZE API
    void build_kernels_ze(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options, std::vector<kernel::ptr> &out) const;
    /// @brief Build kernels through OCL API and repackage to ZE module
    void build_kernels_ocl(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options, std::vector<kernel::ptr> &out) const;
    void init_ocl_builder() const;
    const ze_device &m_device;
    // OCL workaround for legacy devices that does not support ZE compilation
    mutable std::shared_ptr<::cldnn::ocl::ocl_kernel_builder> m_ocl_builder;
    mutable std::shared_ptr<::cldnn::ocl::ocl_device> m_ocl_device;
    mutable std::mutex m_mutex;

};
}  // namespace ze
}  // namespace cldnn

