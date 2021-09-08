// Copyright (C) 2016-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "ocl_memory.hpp"
#include "cldnn/runtime/kernel_args.hpp"
#include "cldnn/runtime/kernel.hpp"

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

    const ocl_kernel_type& get_handle() const { return _compiled_kernel; }
    ocl_kernel_type& get_handle() { return _compiled_kernel; }
    std::shared_ptr<kernel> clone() const override { return std::make_shared<ocl_kernel>(get_handle().clone(), _kernel_id); }
};

}  // namespace ocl
}  // namespace cldnn
