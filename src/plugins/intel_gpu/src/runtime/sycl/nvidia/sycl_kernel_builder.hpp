// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"
#include "../sycl_device.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cldnn {
namespace ocl { class ocl_kernel_builder; }
namespace sycl {
namespace nvidia {

// NVIDIA SYCL kernel builder. OpenCL C kernels are compiled to PTX with clBuildProgram on the NVIDIA OpenCL runtime
// (through ocl::ocl_kernel_builder), and the PTX is wrapped in sycl_ptx_kernel to be launched through CUDA.
class sycl_kernel_builder : public kernel_builder {
public:
    explicit sycl_kernel_builder(const sycl_device& device);

    void build_kernels(const void* src,
                       size_t src_bytes,
                       KernelFormat src_format,
                       const std::string& options,
                       std::vector<kernel::ptr>& out) const override;

private:
    void init_ocl_builder() const;

    const sycl_device& _device;

    // The OpenCL builder is initialized lazily on first use, on the OpenCL device the SYCL device shares.
    mutable std::shared_ptr<::cldnn::ocl::ocl_kernel_builder> _ocl_builder;
    mutable std::once_flag _ocl_builder_once;
};

}  // namespace nvidia
}  // namespace sycl
}  // namespace cldnn
