// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "registry.hpp"

#include "intel_gpu/primitives/fused_mlp.hpp"

#if OV_GPU_WITH_ONEDNN
#    include "impls/onednn/fused_mlp_onednn_graph.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<fused_mlp>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::FusedMLPImplementationManager, shape_types::static_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu

