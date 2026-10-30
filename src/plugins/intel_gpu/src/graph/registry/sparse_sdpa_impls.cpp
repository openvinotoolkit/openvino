// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/sparse_sdpa.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_CM
#    include "impls/cm/sparse_attention/sparse_sdpa.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<sparse_sdpa>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#if OV_GPU_WITH_CM
        OV_GPU_CREATE_INSTANCE_CM(cm::SparseSDPACMImplementationManager, shape_types::any)
#endif
    };

    return impls;
}

}  // namespace ov::intel_gpu
