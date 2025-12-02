// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/vl_sdpa.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_CM
    #include "impls/cm/vl_sdpa_opt.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<vl_sdpa>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_CM(cm::VLSDPAOptImplementationManager, shape_types::any)
    };

    return impls;
}

} // namespace ov::intel_gpu
