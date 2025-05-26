// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "lora_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/lora.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<lora>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::Lora, shape_types::any)
    };

    return impls;
}

}  // namespace ov::intel_gpu
