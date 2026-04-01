// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"
#include "intel_gpu/runtime/device.hpp"

#include "ocl_device.hpp"
#include "ocl_kernel.hpp"

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#error "ocl_kernel_builder.hpp expects OpenCL C++ bindings to throw exceptions"
#endif

// Move this macro to some common header in the future
#define DISCARD_RETURN(func_call) static_cast<void>(func_call)

namespace cldnn {
namespace ocl {

class ocl_kernel_builder : public kernel_builder{
    public:
        ocl_kernel_builder(const ocl_device &device) : m_device(device) {}
        virtual ~ocl_kernel_builder() = default;

        void build_kernels(const void *src,
            size_t src_bytes,
            KernelFormat src_format,
            const std::string &options,
            std::vector<kernel::ptr> &out) const override {
            auto context = m_device.get_context().get();

            cl_program program_handle;
            cl_int err = CL_INVALID_VALUE;
            switch (src_format) {
            case KernelFormat::SOURCE: {
                const char **strings = reinterpret_cast<const char**>(&src);
                const size_t *lenghts = &src_bytes;
                const cl_uint count = 1;
                program_handle = clCreateProgramWithSource(context, count, strings, lenghts, &err);
                break;
            }
            case KernelFormat::NATIVE_BIN: {
                const unsigned char **binaries = reinterpret_cast<const unsigned char**>(&src);
                const size_t *lenghts = &src_bytes;
                const cl_device_id device_id = m_device.get_device().get();
                const cl_uint count = 1;
                program_handle = clCreateProgramWithBinary(context, count, &device_id, lenghts, binaries, nullptr, &err);
                break;
            }
            default:
                OPENVINO_THROW("[GPU] Trying to build kernel from unexpected format");
                break;
            }
            if (err != CL_SUCCESS) {
                OPENVINO_THROW("[GPU] Failed to create program during kernel build process");
            }
            cl::Program program(program_handle);
            cl::vector<cl::Kernel> kernels;
            try {
                // We can safely ignore return values here as those function should throw in case of errors
                DISCARD_RETURN(program.build({m_device.get_device()}, options.c_str()));
                DISCARD_RETURN(program.createKernels(&kernels));
            } catch (const cl::BuildError& err) {
                GPU_DEBUG_INFO << "-------- Kernel build error" << std::endl;
                auto log = err.getBuildLog();
                for (auto &e : log) {
                    GPU_DEBUG_INFO << e.second;
                }
                GPU_DEBUG_INFO << "-------- End of Kernel build error" << std::endl;
                OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
            } catch (const cl::Error& err) {
                OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
            }
            for (auto& k : kernels) {
                const auto &entry_point = k.getInfo<CL_KERNEL_FUNCTION_NAME>();
                out.push_back(std::make_shared<ocl::ocl_kernel>(ocl::ocl_kernel_type(k, m_device.get_usm_helper()), entry_point));
            }
    }

    private:
        const ocl_device &m_device;
};
}  // namespace ocl
}  // namespace cldnn

#undef DISCARD_RETURN
