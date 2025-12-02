// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/convolution_onednn.hpp"
#endif
#if OV_GPU_WITH_OCL
    #include "impls/ocl/convolution.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<convolution>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::ConvolutionImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ConvolutionImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ConvolutionImplementationManager, shape_types::dynamic_shape,
            [](const cldnn::program_node& node){
                if (node.can_use(impl_types::onednn))
                    return false;
                return node.as<convolution>().use_explicit_padding();
        })
    };

    return impls;
}

}  // namespace ov::intel_gpu
