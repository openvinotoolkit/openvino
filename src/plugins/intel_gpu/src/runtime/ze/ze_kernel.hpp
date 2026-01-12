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
    static void create_kernels_from_module(std::shared_ptr<ze_module_holder> module, std::vector<kernel::ptr> &out) {
        ze_module_handle_t module_handle = module->get_module_handle();
        uint32_t kernel_count = 0;
        OV_ZE_EXPECT(zeModuleGetKernelNames(module_handle, &kernel_count, nullptr));
        std::vector<const char*> kernel_names(kernel_count);
        // Specification does not mention who is responsible for the returned pointers
        // Assume Level Zero owns the pointers and they will remain valid as long as the module resource
        OV_ZE_EXPECT(zeModuleGetKernelNames(module_handle, &kernel_count, kernel_names.data()));

        ze_kernel_flags_t flags = 0;
        ze_kernel_desc_t kernel_desc = {
            ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, flags, nullptr};
        for (auto name_cstr : kernel_names) {
            auto name = std::string(name_cstr);
            // L0 returns Intel_Symbol_Table_Void_Program that does not correspond to actual kernel
            if (name == "Intel_Symbol_Table_Void_Program") {
                continue;
            }
            kernel_desc.pKernelName = name_cstr;
            ze_kernel_handle_t kernel_handle;
            OV_ZE_EXPECT(zeKernelCreate(module_handle, &kernel_desc, &kernel_handle));
            auto kernel_holder = std::make_shared<ze_kernel_holder>(kernel_handle, module);
            out.push_back(std::make_shared<ze_kernel>(kernel_holder, name));
        }
    }

    ze_kernel(std::shared_ptr<ze_kernel_holder> kernel, const std::string& kernel_id)
        : m_kernel(kernel)
        , m_kernel_id(kernel_id) { }

    ze_kernel_handle_t get_kernel_handle() const { return m_kernel->get_kernel_handle(); }
    ze_module_handle_t get_module_handle() const { return m_kernel->get_module()->get_module_handle(); }
    std::string get_id() const override { return m_kernel_id; }

    std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const override {
        if (reuse_kernel_handle) {
            return std::make_shared<ze_kernel>(m_kernel, m_kernel_id);
        } else {
            ze_kernel_handle_t cloned_handle;
            ze_module_handle_t module_handle = get_module_handle();
            ze_kernel_desc_t descriptor;
            descriptor.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
            descriptor.pNext = nullptr;
            descriptor.flags = 0;
            descriptor.pKernelName = m_kernel_id.c_str();
            OV_ZE_EXPECT(zeKernelCreate(module_handle, &descriptor, &cloned_handle));
            auto kernel_holder = std::make_shared<ze_kernel_holder>(cloned_handle, m_kernel->get_module());
            return std::make_shared<ze_kernel>(kernel_holder, m_kernel_id);
        }
    }

    virtual bool is_same(const kernel &other) const override {
        auto other_ptr = dynamic_cast<const ze_kernel*>(&other);
        if (other_ptr == nullptr) {
            return false;
        }
        return get_kernel_handle() == other_ptr->get_kernel_handle();
    }

    std::vector<uint8_t> get_binary() const override {
        size_t binary_size = 0;
        ze_module_handle_t module_handle = get_module_handle();
        OV_ZE_EXPECT(zeModuleGetNativeBinary(module_handle, &binary_size, nullptr));

        std::vector<uint8_t> binary(binary_size);
        OV_ZE_EXPECT(zeModuleGetNativeBinary(module_handle, &binary_size, binary.data()));

        return binary;
    }

    std::string get_build_log() const override {
        ze_module_build_log_handle_t build_log_handle = m_kernel->get_module()->get_build_log_handle();
        size_t log_size = 0;
        OV_ZE_EXPECT(zeModuleBuildLogGetString(build_log_handle, &log_size, nullptr));

        std::string log(log_size, ' ');
        OV_ZE_EXPECT(zeModuleBuildLogGetString(build_log_handle, &log_size, log.data()));
        return log;
    }

private:
    std::shared_ptr<ze_kernel_holder> m_kernel;
    std::string m_kernel_id;
};

}  // namespace ze
}  // namespace cldnn
