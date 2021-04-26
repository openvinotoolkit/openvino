// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_factory.hpp"

namespace cldnn {

#ifdef CLDNN_WITH_SYCL
namespace sycl {
std::shared_ptr<kernel> create_sycl_kernel(engine& engine, cl_context context, cl_kernel kernel, std::string  entry_point);
}
#endif

#ifdef CLDNN_WITH_OCL
namespace ocl {
std::shared_ptr<kernel> create_ocl_kernel(engine& engine, cl_context context, cl_kernel kernel, std::string  entry_point);
}  // namespace ocl
#endif

namespace kernels_factory {

std::shared_ptr<kernel> create(engine& engine, cl_context context, cl_kernel kernel, std::string  entry_point) {
    switch (engine.type()) {
#ifdef CLDNN_WITH_OCL
        case engine_types::ocl: return ocl::create_ocl_kernel(engine, context, kernel, entry_point);
#endif
#ifdef CLDNN_WITH_SYCL
        case engine_types::sycl: return sycl::create_sycl_kernel(engine, context, kernel, entry_point);
#endif
        default: throw std::runtime_error("Unsupported engine type in kernels_factory::create");
    }
}

}  // namespace kernels_factory
}  // namespace cldnn
