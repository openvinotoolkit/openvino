// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_kernel.hpp"

#include <memory>
#include <vector>



namespace cldnn {
namespace sycl {


sycl_kernel::sycl_kernel(cl::sycl::kernel compiled_kernel, const std::string& kernel_id)
    : _compiled_kernel(compiled_kernel)
    , _kernel_id(kernel_id) {

}

}  // namespace sycl
}  // namespace cldnn
