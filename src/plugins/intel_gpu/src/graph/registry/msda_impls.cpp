// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/ocl_v2/msda_opt.hpp"
#include "intel_gpu/primitives/msda.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<msda>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MSDAOptImplementationManager, shape_types::static_shape)};

    return impls;
}

}  // namespace ov::intel_gpu