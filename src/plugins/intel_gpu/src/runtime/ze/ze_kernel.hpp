// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel.hpp"
#include "openvino/core/except.hpp"
#include "ze_common.hpp"
#include "ze_kernel_holder.hpp"

#include <memory>

namespace cldnn {
namespace ze {

class ze_kernel : public kernel {
public:
    ze_kernel(std::shared_ptr<ze_kernel_holder> kernel, const std::string& kernel_id)
        : m_kernel(kernel)
        , m_kernel_id(kernel_id) { }

    ze_kernel_handle_t get_kernel() { return m_kernel->get_kernel(); }
    ze_module_handle_t get_module() { return m_kernel->get_module(); }
    std::string get_id() const override { return m_kernel_id; }

    std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const override {
        if (reuse_kernel_handle) {
            return std::make_shared<ze_kernel>(m_kernel, m_kernel_id);
        } else {
            ze_kernel_handle_t cloned_handle;
            ze_module_handle_t module_handle = m_kernel->get_module();
            ze_kernel_desc_t descriptor;
            descriptor.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
            descriptor.pNext = nullptr;
            descriptor.flags = 0;
            descriptor.pKernelName = m_kernel_id.c_str();
            OV_ZE_EXPECT(zeKernelCreate(module_handle, &descriptor, &cloned_handle));
            return std::make_shared<ze_kernel>(cloned_handle, module_handle, m_kernel_id);
        }
    }

    std::vector<uint8_t> get_binary() const override {
        size_t binary_size = 0;
        ze_module_handle_t module_handle = m_kernel->get_module();
        OV_ZE_EXPECT(zeModuleGetNativeBinary(module_handle, &binary_size, nullptr));

        std::vector<uint8_t> binary(binary_size);
        OV_ZE_EXPECT(zeModuleGetNativeBinary(module_handle, &binary_size, binary.data()));

        return binary;
    }
private:
    std::shared_ptr<ze_kernel_holder> m_kernel;
    std::string m_kernel_id;
};

}  // namespace ze
}  // namespace cldnn
