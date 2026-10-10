// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "registry.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"


#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/fully_connected_onednn.hpp"
#endif
// The TernOCL int2 impl builds its programs on the OpenCL engine's context.
#if OV_GPU_WITH_OCL && defined(OV_GPU_WITH_OCL_RT)
#    include "impls/ocl_v2/ternocl_int2/fully_connected_ternocl_int2.hpp"
#    define OV_GPU_CREATE_INSTANCE_TERNOCL(...) OV_GPU_CREATE_INSTANCE_OCL(__VA_ARGS__)
#else
#    define OV_GPU_CREATE_INSTANCE_TERNOCL(...)
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<fully_connected>::get_implementations() {
    static const auto ocl_supports_weights_layout = [](const program_node& node) {
        return node.as<fully_connected>().get_primitive()->weights_transposed;
    };

    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_TERNOCL(cldnn::ocl::TernoclInt2FCImplementationManager, shape_types::dynamic_shape)
            OV_GPU_CREATE_INSTANCE_TERNOCL(cldnn::ocl::TernoclInt2FCImplementationManager, shape_types::static_shape)
                OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::FullyConnectedImplementationManager, shape_types::static_shape)
                    OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::static_shape, ocl_supports_weights_layout)
                        OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::dynamic_shape, [](const program_node& node) {
                            if (node.can_use(impl_types::onednn))
                                return false;
                            if (!ocl_supports_weights_layout(node))
                                return false;
                            return node.get_output_pshape().size() <= 3;
                        })};

    return impls;
}

}  // namespace ov::intel_gpu
