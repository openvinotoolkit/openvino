// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "registry.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "impls/ocl_v2/gemm/fc_compressed_generate_opt.hpp"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/fully_connected_onednn.hpp"
#endif

#if OV_GPU_WITH_CM
    #include "impls/cm/fc_compressed_generate_opt.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<fully_connected>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::FullyConnectedImplementationManager, shape_types::static_shape)
        // CM GEMV for LLM generate phase — compiled via CM compiler for potentially tighter ISA.
        // Same constraints as OCL version below; additionally requires CM JIT + IMMAD support.
        OV_GPU_CREATE_INSTANCE_CM(cm::FCCompressedGenerateOptCM, shape_types::static_shape)
        // Optimised WOQ GEMV for LLM generate phase (M=1, static shape, f16-act + int4-weight).
        // validate_impl: compressed_weights, f16 activation, u4/i4 weight, f16 scale.
        // support_shapes: M==1 + K%128 == 0 checked at runtime by the impl-pool builder.
        OV_GPU_CREATE_INSTANCE_OCL(ocl::FCCompressedGenerateOpt, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::dynamic_shape,
            [](const program_node& node) {
                if (node.can_use(impl_types::onednn))
                    return false;
                return node.get_output_pshape().size() <= 3;
        })
    };

    return impls;
}

}  // namespace ov::intel_gpu
