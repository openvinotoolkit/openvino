// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_factory.hpp"

namespace cldnn {
namespace ocl {
std::shared_ptr<kernel> create_ocl_kernel(engine& engine, cl_context context, cl_kernel kernel, std::string  entry_point);
}  // namespace ocl

namespace kernels_factory {

std::shared_ptr<kernel> create(engine& engine, cl_context context, cl_kernel kernel, std::string  entry_point) {
    switch (engine.type()) {
        case engine_types::ocl: return ocl::create_ocl_kernel(engine, context, kernel, entry_point);
        default: throw std::runtime_error("Unsupported engine type in kernels_factory::create");
    }
}

}  // namespace kernels_factory
}  // namespace cldnn
