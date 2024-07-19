// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_inst.h"
#include "impls/registry/implementation_manager.hpp"

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

// Return true if one of blocked axes (b or f) is reduced and one of spatial axes is NOT reduced
inline bool is_reduce_blocked_axes(reduce_node const& node) {
    auto prim = node.get_primitive();
    auto reduce_axes = prim->axes;
    auto input_layout = node.get_input_layout();
    auto num_spatial = format::spatial_num(node.get_output_layout().format);
    auto dims = node.get_output_layout().format.dimension();

    // Check if it reduces all spatial axes
    bool feature_axis_is_only_remaining = true;
    for (size_t idx_spatial = (dims - num_spatial); idx_spatial < dims; idx_spatial++) {
        if (count(reduce_axes.begin(), reduce_axes.end(), idx_spatial) == 0) {
            feature_axis_is_only_remaining = false;
            break;
        }
    }

    if (input_layout.is_static() &&
        (count(reduce_axes.begin(), reduce_axes.end(), 1) > 0 ||
        (count(reduce_axes.begin(), reduce_axes.end(), 0) > 0))) {
        if (!feature_axis_is_only_remaining)
            return true;
    }

    return false;
}

struct ReduceImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ReduceImplementationOnednn")
    ReduceImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<reduce>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad)
            return false;

        const auto& reduce_node = node.as<reduce>();
        auto preferred_format = reduce_node.get_preferred_input_fmt(0);

        auto reduce_prim = reduce_node.get_primitive();
        const auto& in_layout = reduce_node.get_input_layout(0);
        const auto& out_layout = reduce_node.get_output_layout(0);
        auto in_dt = in_layout.data_type;
        auto out_dt = out_layout.data_type;

        if (in_dt == data_types::f32 && out_dt == data_types::f32)
            return false;

        static const std::vector<format::type> supported_formats = {
            format::bfyx,
            format::bfzyx,
            format::bfwzyx,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv32,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
        };

        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats))
            return false;

        // oneDNN reduction currently does not support logical_and, logical_or, log_sum and log_sum_exp.
        switch (reduce_prim->mode) {
            case reduce_mode::mean:
            case reduce_mode::max:
            case reduce_mode::min:
            case reduce_mode::sum:
            case reduce_mode::prod:
                break;
            case reduce_mode::sum_square:
            case reduce_mode::l1:
            case reduce_mode::l2:
                // modes have a limitation of data type
                if (one_of(in_dt, {data_types::f16, data_types::f32}))
                    break;
            default:
                return false;
        }

        // redundant reduce is not acceptable on oneDNN reduction
        if (out_layout == in_layout) {
            return false;
        }

        // oneDNN reduction selects ref kernel for simple formats(bfyx..) which has perf regression with a decent tensor size.
        if (format::is_simple_data_format(preferred_format))
            return false;

        // Onednn reduction does NOT support reordering of unreduced-axes.
        // Currently, an Onednn reduce layer which contains reduction of blocked axes(b-f) is expected to select planar format.
        if (reduce_prim->keep_dims == false && is_reduce_blocked_axes(node))
            return false;

        return ImplementationManager::validate(node);
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    bool support_shapes(const kernel_impl_params& params) const override {
        return true;
    }
};

}  // namespace onednn
}  // namespace cldnn
