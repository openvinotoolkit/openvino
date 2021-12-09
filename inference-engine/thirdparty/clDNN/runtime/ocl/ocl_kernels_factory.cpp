// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_kernel.hpp"
#include "ocl_engine.hpp"
#include "kernels_factory.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace ocl {

std::shared_ptr<kernel> create_ocl_kernel(engine& engine, cl_context /* context */, cl_kernel kernel, std::string entry_point) {
    // Retain kernel to keep it valid
    cl::Kernel k(kernel, true);
    ocl_engine& cl_engine = dynamic_cast<ocl_engine&>(engine);
    return std::make_shared<ocl::ocl_kernel>(ocl::ocl_kernel_type(k, cl_engine.get_usm_helper()), entry_point);
}

}  // namespace ocl
}  // namespace cldnn
