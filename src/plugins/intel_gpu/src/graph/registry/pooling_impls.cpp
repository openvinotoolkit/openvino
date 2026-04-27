// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/pooling.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/pooling_onednn.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<pooling>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::PoolingImplementationManager, shape_types::static_shape, [](const program_node& node) {
            const auto& in_layout = node.get_input_layout(0);
            const auto& out_layout = node.get_output_layout(0);
            // Disable this case due to sporadic hang for the following case:
            // onednn_verbose,primitive,exec,gpu:0,pooling,jit:ir,forward_inference,src_u8::blocked:acdb::f0 dst_u8::blocked:abcd::f0
            // ws_undef::undef:::,attr-scratchpad:user attr-post-ops:eltwise_linear:1.52456,alg:pooling_avg_include_padding,
            // mb1ic96_ih56oh28kh2sh2dh0ph0_iw56ow28kw2sw2dw0pw0,0.0400391
            // issue: 12579
            if (in_layout.format == format::byxf && out_layout.format == format::bfyx)
                return false;
            // Disable oneDNN JIT pooling for int8/uint8 with b_fs_yx_fsv16 blocked format
            // on Xe2+ hardware. The oneDNN JIT kernel (jit:ir) has out-of-bounds memory
            // accesses with this format that cause CL_OUT_OF_RESOURCES crashes on Xe2+
            // (which enforces strict OOB checking). Pre-Xe2 hardware silently returns
            // zero for OOB reads, so the issue is benign there.
            // Fall back to the OCL pooling implementation on affected architectures.
            if (node.get_program().get_engine().get_device_info().arch >= gpu_arch::xe2 &&
                (in_layout.format == format::b_fs_yx_fsv16 || in_layout.format == format::b_fs_zyx_fsv16) &&
                (in_layout.data_type == data_types::i8 || in_layout.data_type == data_types::u8)) {
                return false;
            }
            return true;
        })
        OV_GPU_GET_INSTANCE_OCL(pooling, shape_types::static_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
