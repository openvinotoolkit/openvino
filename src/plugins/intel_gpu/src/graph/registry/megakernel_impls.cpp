// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/megakernel.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#ifdef OV_GPU_MEGAKERNEL_IMPL_LINKED
#    include "impls/ocl_v2/megakernel_interface/megakernel.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>&
Registry<megakernel>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#ifdef OV_GPU_MEGAKERNEL_IMPL_LINKED
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MegaKernelImpl, shape_types::any)
#endif
    };
    return impls;
}

}  // namespace ov::intel_gpu
