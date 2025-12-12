// Copyright (C) 2016-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel_builder.hpp"
#include "intel_gpu/runtime/device.hpp"

#include "ze_device.hpp"
#include "ze_kernel.hpp"
#include "ze_common.hpp"

namespace cldnn {
namespace ze {

class ze_kernel_builder : public kernel_builder{
    public:
        ze_kernel_builder(const ze_device &device) : m_device(device) {}

    void build_kernels(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options, std::vector<kernel::ptr> &out) const override {
        ze_module_desc_t module_desc = {
            ZE_STRUCTURE_TYPE_MODULE_DESC,
            nullptr,
            ZE_MODULE_FORMAT_NATIVE,
            src_bytes,
            reinterpret_cast<const uint8_t *>(src),
            options.c_str(),
            nullptr // specialization constants
        };
        switch (src_format) {
        case KernelFormat::SOURCE: {
            module_desc.format = ze_module_format_oclc;
            break;
        }
        case KernelFormat::NATIVE_BIN: {
            module_desc.format = ZE_MODULE_FORMAT_NATIVE;
            break;
        }
        default:
            OPENVINO_THROW("[GPU] Trying to build kernel from unexpected format");
            break;
        }
        ze_module_handle_t module_handle;
        ze_module_build_log_handle_t log_handle;
        ze_result_t build_result = zeModuleCreate(m_device.get_context(), m_device.get_device(), &module_desc, &module_handle, &log_handle);
        if (build_result != ZE_RESULT_SUCCESS) {
            size_t log_size = 0;
            OV_ZE_EXPECT(zeModuleBuildLogGetString(log_handle, &log_size, nullptr));
            std::string log(log_size, ' ');
            OV_ZE_EXPECT(zeModuleBuildLogGetString(log_handle, &log_size, log.data()));
            OV_ZE_EXPECT(zeModuleBuildLogDestroy(log_handle));
            GPU_DEBUG_INFO << "-------- Kernel build error" << std::endl;
            GPU_DEBUG_INFO << log << std::endl;
            GPU_DEBUG_INFO << "-------- End of Kernel build error" << std::endl;
            OPENVINO_THROW("[GPU] Failed to build module");
        }
        auto module_holder = std::make_shared<ze_module_holder>(module_handle, log_handle);
        ze_kernel::create_kernels_from_module(module_holder, out);
    }

    private:
        const ze_device &m_device;
};
}  // namespace ze
}  // namespace cldnn

