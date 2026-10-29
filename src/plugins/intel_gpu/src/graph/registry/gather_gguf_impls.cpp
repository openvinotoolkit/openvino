// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/gather_gguf.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/gguf/gather_gguf.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<gather_gguf>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GatherGGUFRef, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GatherGGUFRef, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
