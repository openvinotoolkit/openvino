// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include "ze_memory.hpp"
#include "cldnn/runtime/kernel_args.hpp"
#include "cldnn/runtime/kernel.hpp"

#include <memory>
#include <vector>

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

    const ze_kernel_handle_t& get_handle() const { return _compiled_kernel; }
    ze_kernel_handle_t& get_handle() { return _compiled_kernel; }
    std::shared_ptr<kernel> clone() const override {
        ze_kernel_handle_t cloned_handle;
        ze_kernel_desc_t descriptor;
        descriptor.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        descriptor.pNext = nullptr;
        descriptor.flags = 0;
        descriptor.pKernelName = _kernel_id.c_str();
        ZE_CHECK(zeKernelCreate(_module, &descriptor, &cloned_handle));
        return std::make_shared<ze_kernel>(cloned_handle, _module, _kernel_id);
    }
};

}  // namespace ze
}  // namespace cldnn
