// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_SYCL
    #include "impls/sycl/eltwise.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<eltwise>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#if OV_GPU_WITH_SYCL
        OV_GPU_CREATE_INSTANCE_SYCL(cldnn::sycl::EltwiseImplementationManager, shape_types::static_shape)
#endif
        OV_GPU_GET_INSTANCE_OCL(eltwise, shape_types::static_shape, not_in_shape_flow())
        OV_GPU_GET_INSTANCE_OCL(eltwise, shape_types::dynamic_shape, not_in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(eltwise, shape_types::static_shape, in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(eltwise, shape_types::dynamic_shape, in_shape_flow())
    };

    return impls;
}

}  // namespace ov::intel_gpu
