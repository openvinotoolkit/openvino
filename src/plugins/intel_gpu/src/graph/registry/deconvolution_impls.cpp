// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/deconvolution_onednn.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<deconvolution>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::DeconvolutionImplementationManager, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(deconvolution, shape_types::static_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
