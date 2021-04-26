// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl_common.hpp"
#include "sycl_memory.hpp"
#include "cldnn/runtime/kernel_args.hpp"
#include "cldnn/runtime/kernel.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace sycl {

class sycl_kernel : public kernel {
    cl::sycl::kernel _compiled_kernel;
    std::string _kernel_id;

public:
    sycl_kernel(cl::sycl::kernel compiled_kernel, const std::string& kernel_id);

    // sycl_kernel(const sycl_kernel& other)
    //     : _compiled_kernel(other._compiled_kernel)
    //     , _kernel_id(other._kernel_id) {}

    // sycl_kernel& operator=(const sycl_kernel& other) {
    //     if (this == &other) {
    //         return *this;
    //     }

    //     _kernel_id = other._kernel_id;
    //     _compiled_kernel = other._compiled_kernel;

    //     return *this;
    // }

    std::shared_ptr<kernel> clone() const override { return std::make_shared<sycl_kernel>(get_handle(), _kernel_id); }
    const cl::sycl::kernel& get_handle() const { return _compiled_kernel; }
};

}  // namespace sycl
}  // namespace cldnn
