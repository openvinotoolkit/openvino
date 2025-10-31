// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/msda.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#include "impls/ocl_v2/msda_opt.hpp"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<msda>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MSDAOptImplementationManager, shape_types::any)
    };

    return impls;
}

} // namespace ov::intel_gpu