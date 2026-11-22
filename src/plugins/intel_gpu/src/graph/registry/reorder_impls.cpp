// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "primitive_inst.h"
#include "reorder_inst.h"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/reorder_onednn.hpp"
#endif
#if OV_GPU_WITH_OCL
    #include "impls/ocl/reorder.hpp"
#endif
// Packs the weights of the TernOCL int2 FullyConnected impl, so it is registered wherever that impl is.
#if OV_GPU_WITH_OCL && defined(OV_GPU_WITH_OCL_RT)
#    include "impls/ocl_v2/ternocl_int2/ternocl_int2_weights_reorder.hpp"
#    define OV_GPU_CREATE_INSTANCE_TERNOCL(...) OV_GPU_CREATE_INSTANCE_OCL(__VA_ARGS__)
#else
#    define OV_GPU_CREATE_INSTANCE_TERNOCL(...)
#endif

namespace ov::intel_gpu {

using namespace cldnn;

// Formats listed here gate the dynamic-shape OCL reorder path: both in and out
// layouts must be in this set for the dynamic ImplementationManager to be offered.
// Each entry requires a dynamic-capable kernel (Ref covers all; bfyx_to_blocked_format
// handles bfyx/bfzyx -> blocked; fsv handles blocked <-> blocked). Entries below
// match what reorder_gpu_optimization.dynamic_* tests exercise; extend only when a
// production network needs a new dynamic reorder path + a matching optimized kernel.
static std::vector<format> supported_dyn_formats = {
    format::bfyx,
    format::bfzyx,
    format::bfwzyx,
    format::byxf,
    format::b_fs_yx_fsv4,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::b_fs_zyx_fsv16,
    format::b_fs_zyx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_zyx_bsv16_fsv32,
};

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<reorder>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_TERNOCL(ocl::TernoclInt2WeightsReorderImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::ReorderImplementationManager, shape_types::static_shape, not_in_shape_flow())
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ReorderImplementationManager, shape_types::static_shape, not_in_shape_flow())
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ReorderImplementationManager, shape_types::dynamic_shape,
            [](const program_node& node) {
                const auto& in_layout = node.get_input_layout(0);
                const auto& out_layout = node.get_output_layout(0);
                if (!one_of(in_layout.format, supported_dyn_formats) || !one_of(out_layout.format, supported_dyn_formats))
                    return false;
                // CPU impl does not support b_fs_yx_fsv16 format, so prefer CPU impl only
                // when both input and output formats are simple (CPU-compatible)
                return !node.is_in_shape_of_subgraph() || !format::is_simple_data_format(out_layout.format)
                    || !format::is_simple_data_format(in_layout.format);
            })
        OV_GPU_GET_INSTANCE_CPU(reorder, shape_types::static_shape, in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(reorder, shape_types::dynamic_shape, in_shape_flow())
    };

    return impls;
}

}  // namespace ov::intel_gpu
