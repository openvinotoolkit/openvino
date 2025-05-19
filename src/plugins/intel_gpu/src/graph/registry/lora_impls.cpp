// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "lora_inst.h"
#include "impls/ocl/lora.hpp"

#if OV_GPU_WITH_CM
    #include "impls/cm/xetla_lora.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<lora>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_CM(cm::LoRAImplementationManager, shape_types::any)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::LoraImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::LoraImplementationManager, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
