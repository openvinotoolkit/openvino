// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "registry.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "impls/ocl_v2/gemm/fc_compressed_generate_opt.hpp"
#include "impls/ocl_v2/gguf/fc_gguf_opt.hpp"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/fully_connected_onednn.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<fully_connected>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::FullyConnectedImplementationManager, shape_types::static_shape)
        // Optimised WOQ GEMV for LLM generate phase (M=1, static shape, f16-act + int4-weight).
        // validate_impl: compressed_weights, f16 activation, u4/i4 weight, f16 scale.
        // support_shapes: M==1 + K%128 == 0 checked at runtime by the impl-pool builder.
        OV_GPU_CREATE_INSTANCE_OCL(ocl::FCCompressedGenerateOpt, shape_types::static_shape)
        // Native GGUF weight-only-quantised FC (covers all GGUF block formats via one manager; only
        // the baseline qwen3 formats have a kernel in this release — see ocl::FCGGUFOpt). Registered
        // shape-agnostic ("any"): GGUF models are dynamic and this is the sole impl able to read GGUF
        // blocks, so it binds the dynamic node at compile time (shape-agnostic kernel reading rows from
        // shape_info) and is re-specialised per concrete shape at runtime.
        OV_GPU_CREATE_INSTANCE_OCL(ocl::FCGGUFOpt, shape_types::any)
        OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::dynamic_shape,
            [](const program_node& node) {
                // GGUF weights are only consumable by ocl::FCGGUFOpt (a static-shape impl picked at
                // runtime per concrete shape); the legacy kernel-selector FC cannot read GGUF blocks
                // and would throw in to_weights_type. Never bind it to a GGUF-weight FC.
                if (ov::element::is_gguf_block(node.get_input_layout(1).data_type))
                    return false;
                if (node.can_use(impl_types::onednn))
                    return false;
                return node.get_output_pshape().size() <= 3;
        })
    };

    return impls;
}

}  // namespace ov::intel_gpu
