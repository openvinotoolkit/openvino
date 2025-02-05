// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/concatenation_onednn.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<concatenation>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::ConcatenationImplementationManager, shape_types::static_shape, not_in_shape_flow())
        OV_GPU_GET_INSTANCE_OCL(concatenation, shape_types::static_shape, not_in_shape_flow())
        OV_GPU_GET_INSTANCE_OCL(concatenation, shape_types::dynamic_shape, not_in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(concatenation, shape_types::static_shape, in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(concatenation, shape_types::dynamic_shape, in_shape_flow())
    };

    return impls;
}

}  // namespace ov::intel_gpu
