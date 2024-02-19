// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_factory.hpp"

#include <memory>
#include <string>

namespace cldnn {
namespace ocl {

std::shared_ptr<kernel> create_ocl_kernel(engine& engine, cl_context /* context */, cl_kernel kernel, std::string entry_point);

}  // namespace ocl
}  // namespace cldnn
