// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_kernel.hpp"
#include "kernels_factory.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace ocl {

std::shared_ptr<kernel> create_ocl_kernel(engine& engine, cl_context /* context */, cl_kernel kernel, std::string entry_point) {
    // Retain kernel to keep it valid
    cl::Kernel k(kernel, true);
    return std::make_shared<ocl::ocl_kernel>(ocl::ocl_kernel_type(k, engine.use_unified_shared_memory()), entry_point);
}

}  // namespace kernels_factory
}  // namespace cldnn
