// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel.hpp"
#include "openvino/core/except.hpp"
#include "ze_common.hpp"

#include <memory>

namespace cldnn {
namespace ze {

class ze_kernel : public kernel {
    ze_kernel_handle_t _compiled_kernel;
    ze_module_handle_t _module;
    std::string _kernel_id;

public:
    ze_kernel(ze_kernel_handle_t compiled_kernel, ze_module_handle_t module, const std::string& kernel_id)
        : _compiled_kernel(compiled_kernel)
        , _module(module)
        , _kernel_id(kernel_id) { }

    ~ze_kernel() {
        zeKernelDestroy(_compiled_kernel);
    }

    const ze_kernel_handle_t& get_handle() const { return _compiled_kernel; }
    ze_kernel_handle_t& get_handle() { return _compiled_kernel; }
    std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const override {
        if (reuse_kernel_handle) {
            return std::make_shared<ze_kernel>(_compiled_kernel, _module, _kernel_id);
        } else {
            ze_kernel_handle_t cloned_handle;
            ze_kernel_desc_t descriptor;
            descriptor.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
            descriptor.pNext = nullptr;
            descriptor.flags = 0;
            descriptor.pKernelName = _kernel_id.c_str();
            ZE_CHECK(zeKernelCreate(_module, &descriptor, &cloned_handle));
            return std::make_shared<ze_kernel>(cloned_handle, _module, _kernel_id);
        }
    }

    std::string get_id() const override { return _kernel_id; }

    std::vector<uint8_t> get_binary() const override {
        size_t binary_size = 0;
        ZE_CHECK(zeModuleGetNativeBinary(_module, &binary_size, nullptr));

        std::vector<uint8_t> binary(binary_size);
        ZE_CHECK(zeModuleGetNativeBinary(_module, &binary_size, &binary[0]));

        return binary;
    }
};

}  // namespace ze
}  // namespace cldnn
