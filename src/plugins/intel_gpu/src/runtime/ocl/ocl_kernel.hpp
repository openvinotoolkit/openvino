// Copyright (C) 2016-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "ocl_memory.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/kernel.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace ocl {

class ocl_kernel : public kernel {
    ocl_kernel_type _compiled_kernel;
    std::string _kernel_id;

public:
    ocl_kernel(ocl_kernel_type compiled_kernel, const std::string& kernel_id)
        : _compiled_kernel(compiled_kernel)
        , _kernel_id(kernel_id) { }

    std::string get_id() const override { return _kernel_id; }
    const ocl_kernel_type& get_handle() const { return _compiled_kernel; }
    ocl_kernel_type& get_handle() { return _compiled_kernel; }
    std::shared_ptr<kernel> clone() const override {
#ifdef GPU_DEBUG_CONFIG
        if (cldnn::debug_configuration::get_instance()->check_kernels_properties >= 2) {
            GPU_DEBUG_TRACE_DETAIL << "Clone " << _kernel_id << " kernel with properties: " << get_properties().to_string() << "\n";
        }
#endif
        return std::make_shared<ocl_kernel>(get_handle().clone(), _kernel_id);
    }

#ifdef GPU_DEBUG_CONFIG
    kernel_properties get_properties() const override {
        const auto kernel = get_handle();
        const auto ctx = kernel.getInfo<CL_KERNEL_CONTEXT>();
        const auto devices = ctx.getInfo<CL_CONTEXT_DEVICES>();

        kernel_properties properties;
        if (devices.size() == 1) {
            const auto device = devices[0];

            cl_int err = CL_SUCCESS;
            cl_ulong local_mem_size = kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device, &err);
            if (err == CL_SUCCESS) {
                properties.local_mem_size = local_mem_size;
            }

            cl_ulong spill_mem_size = kernel.getWorkGroupInfo<CL_KERNEL_SPILL_MEM_SIZE_INTEL>(device, &err);
            if (err == CL_SUCCESS) {
                properties.spill_mem_size = spill_mem_size;
            }

            cl_ulong private_mem_size = kernel.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device, &err);
            if (err == CL_SUCCESS) {
                properties.private_mem_size = private_mem_size;
            }
        }

        return properties;
    }
#endif
};

}  // namespace ocl
}  // namespace cldnn
