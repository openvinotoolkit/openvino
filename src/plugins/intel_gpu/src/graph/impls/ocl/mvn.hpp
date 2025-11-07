// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry/implementation_manager.hpp"
#include "program_node.h"
#include "mvn_inst.h"

#include <memory>

namespace cldnn {

namespace ocl {

struct MVNImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::mvn")
    MVNImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        auto input_layout = node.get_input_layout(0);
        auto input_pshape = input_layout.get_partial_shape();
        auto prim = node.as<mvn>().get_primitive();
        if (input_layout.data_type == data_types::f32 || input_layout.data_type == data_types::f16) {
            in_fmts[0] = format::get_default_format(input_layout.get_rank());
            out_fmts[0] = format::get_default_format(node.get_output_layout(0).get_rank());
        } else if (prim->requires_alignment(input_pshape)) {
            auto block_sizes = format::block_sizes(format::b_fs_yx_fsv16);
            auto blocked_axis = block_sizes[0].first;
            auto block_size = block_sizes[0].second;
            auto& reduction_axes = prim->reduction_axes;
            
            // Use plain format if:
            // 1. Dynamic shape (can't verify alignment at compile time), OR
            // 2. Static shape that's unaligned and the blocked axis is NOT reduced
            bool is_unaligned = input_pshape[blocked_axis].is_dynamic() ||
                                (input_pshape[blocked_axis].get_length() % block_size != 0);
            bool axis_not_reduced = std::count(reduction_axes.begin(), reduction_axes.end(), blocked_axis) == 0;
            if (input_layout.is_dynamic() || (is_unaligned && axis_not_reduced)) {
                in_fmts[0] = format::get_default_format(input_layout.get_rank());
                out_fmts[0] = format::get_default_format(node.get_output_layout(0).get_rank());
            }
        }
        return {in_fmts, out_fmts};
    }
};

} // namespace ocl
} // namespace cldnn
