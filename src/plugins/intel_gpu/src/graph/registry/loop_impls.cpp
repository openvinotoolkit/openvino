// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/loop.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_COMMON
    #include "impls/common/loop.hpp"
#endif


namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<loop>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_COMMON(common::LoopImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_COMMON(common::LoopImplementationManager, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
