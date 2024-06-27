// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_optimizer.h"
#include "primitive_inst.h"
#include "program_helpers.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "data_inst.h"
#include "reorder_inst.h"
#include "resample_inst.h"
#include "reshape_inst.h"
#include "arg_max_min_inst.h"
#include "shape_of_inst.h"
#include "condition_inst.h"
#include "strided_slice_inst.h"
#include <sstream>

#include "gemm_inst.h"
#include "deconvolution_inst.h"
#include "fully_connected_inst.h"
#include "non_max_suppression_inst.h"
#include "eltwise_inst.h"
#include "pooling_inst.h"
#include "reduce_inst.h"
#include "one_hot_inst.h"
#include "permute_inst.h"
#include "quantize_inst.h"
#include "mvn_inst.h"
#include "depth_to_space_inst.h"
#include "region_yolo_inst.h"
#include "prior_box_inst.h"
#include "scatter_nd_update_inst.h"
#include "gather_inst.h"
#include "loop_inst.h"
#include "dft_inst.h"
#include "to_string_utils.h"
#include <vector>
#include <memory>
#include <utility>

#include "pass_manager.h"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl.hpp>
#include "impls/onednn/utils.hpp"
#endif

using namespace cldnn;

static size_t get_post_ops_count(const program_node& node) {
    size_t onednn_post_ops_count = 0;
    for (auto& fo : node.get_fused_primitives()) {
       onednn_post_ops_count += fo.f_param->ops_count();
    }

    return onednn_post_ops_count;
}

// Return true if one of blocked axes (b or f) is reduced and one of spatial axes is NOT reduced
static bool is_reduce_blocked_axes(reduce_node const& node) {
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

bool layout_optimizer::onednn_check_data_types_for_pooling(data_types in_dt, data_types out_dt) {
    if (!data_type_traits::is_floating_point(in_dt) && in_dt != out_dt)
            return false;
    if ((in_dt == data_types::i8 || in_dt == data_types::u8) && out_dt != data_types::f32)
        return true;
    if (in_dt == data_types::f16 || out_dt == data_types::f16)
        return true;
    if (out_dt == data_types::f32)
        return true;
    if (in_dt == data_types::i32 || out_dt == data_types::i32)
        return true;
    if ((in_dt == data_types::i8 || out_dt == data_types::i8) || (in_dt == data_types::u8 || out_dt == data_types::u8))
        return true;
    return false;
}

bool layout_optimizer::onednn_check_data_types_for_convolution(data_types in_dt, data_types wei_dt, data_types out_dt) {
    if ((in_dt == data_types::f16 && wei_dt == data_types::f16) &&
        (out_dt == data_types::f16 || out_dt == data_types::f32 || out_dt == data_types::i8 || out_dt == data_types::u8))
        return true;
    if ((in_dt == data_types::i8 || in_dt == data_types::u8) && wei_dt == data_types::i8 &&
        (out_dt == data_types::f32 || out_dt == data_types::i32 || out_dt == data_types::f16 || out_dt == data_types::i8 || out_dt == data_types::u8))
        return true;
    if ((in_dt == data_types::f32 && wei_dt == data_types::f32) &&
        (out_dt == data_types::i8 || out_dt == data_types::u8))
        return true;
    return false;
}

// almost same with onednn_check_data_types_for_convolution.
// removed case
// - in_dt(f16) wei_dt(f16) out_dt(f32)
bool layout_optimizer::onednn_check_data_types_for_deconvolution(data_types in_dt, data_types wei_dt, data_types out_dt) {
    if ((in_dt == data_types::f16 && wei_dt == data_types::f16) &&
        (out_dt == data_types::f16 || out_dt == data_types::i8 || out_dt == data_types::u8))
        return true;
    if ((in_dt == data_types::i8 || in_dt == data_types::u8) && wei_dt == data_types::i8 &&
        (out_dt == data_types::f32 || out_dt == data_types::i32 || out_dt == data_types::f16 || out_dt == data_types::i8 || out_dt == data_types::u8))
        return true;
    if ((in_dt == data_types::f32 && wei_dt == data_types::f32) &&
        (out_dt == data_types::i8 || out_dt == data_types::u8))
        return true;
    return false;
}

bool layout_optimizer::onednn_check_data_types_for_fc_gemm(data_types in_dt, data_types wei_dt, data_types out_dt) {
    if ((in_dt == data_types::f16 && wei_dt == data_types::f16) &&
        (out_dt == data_types::f16 || out_dt == data_types::f32 || out_dt == data_types::i8))
        return true;
    if (in_dt == data_types::f32 && wei_dt == data_types::f32)
        return true;
    if ((in_dt == data_types::i8 || in_dt == data_types::u8) && (wei_dt == data_types::i8) &&
        (out_dt == data_types::i8 || out_dt == data_types::u8 || out_dt == data_types::i32 || out_dt == data_types::f16 || out_dt == data_types::f32))
        return true;
    return false;
}

std::pair<std::shared_ptr<reorder>, bool> reorder_factory::get_reorder(primitive_id src_id,
                                                                       int32_t src_port,
                                                                       const layout& in_layout,
                                                                       const layout& out_layout) {
    if (in_layout == out_layout)
        return std::make_pair(nullptr, true);

    cache_key ckey{ src_id + "." + std::to_string(src_port), out_layout };
    auto itr = _cached_reorders.find(ckey);
    if (itr != _cached_reorders.end())
        return std::make_pair(itr->second, true);

    auto count = _cached_reorders.size();
    std::stringstream ss;
    ss << src_id << "_" << std::to_string(src_port) << "_reorder_" << count;

    auto reorder = std::make_shared<cldnn::reorder>(ss.str(), input_info{src_id, src_port}, out_layout);
    _cached_reorders[ckey] = reorder;

    return std::make_pair(reorder, false);
}

std::pair<std::shared_ptr<reorder>, bool> reorder_factory::get_reorder(primitive_id src_id,
                                                                       const layout& in_layout,
                                                                       const layout& out_layout) {
    return get_reorder(src_id, 0, in_layout, out_layout);
}

std::pair<std::shared_ptr<primitive>, bool> reorder_factory::get_weights_reorder(primitive_id input_id,
                                                                                 std::shared_ptr<WeightsReorderParams> reorder_params) {
    OPENVINO_ASSERT(reorder_params != nullptr, "[GPU] WeightsReorderParams is not initialized.");

    cache_key ckey{ input_id, reorder_params->get_output_layout(), false };
    auto itr = _cached_reorders.find(ckey);
    if (itr != _cached_reorders.end()) {
        return std::make_pair(itr->second, true);
    } else {
        auto count = _cached_reorders.size();
        std::string reorder_id = input_id + "_weights_reorder_" + std::to_string(count);

        auto reorder = std::make_shared<cldnn::reorder>(reorder_id, input_id, reorder_params);
        _cached_reorders[ckey] = reorder;
        return std::make_pair(reorder, false);
    }
}

bool layout_optimizer::is_format_supported(program_node& node, format::type fmt) {
    if (node.is_type<fully_connected>() && fmt == format::byxf)
        return false;

    if (node.is_type<mvn>() && fmt == format::b_fs_yx_fsv16 &&
        node.get_input_layout(0).data_type != data_types::i8 &&
        node.get_input_layout(0).data_type != data_types::u8)
        return false;

    if (node.is_type<input_layout>())
        return node.get_output_layout().format == fmt;

    if (!_forcing_map.empty() && _forcing_map.count(node.id()))
        return _forcing_map.at(node.id()).first == fmt;

    auto prev_layout = node.get_output_layout();
    auto new_layout = prev_layout;
    new_layout.format = fmt;
    node.set_output_layout(new_layout, false);

    auto supported = node.type()->does_possible_implementation_exist(node);

    node.set_output_layout(prev_layout, false);

    return supported;
}

bool layout_optimizer::can_fuse_reorder(program_node& prev, program_node& next, format fmt_prev, format fmt_next) {
    auto prev_simple = fmt_prev == format::bfyx || fmt_prev == format::byxf || fmt_prev == format::yxfb;
    auto next_simple = fmt_next == format::bfyx || fmt_next == format::byxf || fmt_next == format::yxfb;
    auto prev_output_layout = prev.get_output_layout();
    auto next_output_layout = next.get_output_layout();
    auto prev_dt = prev.get_output_layout().data_type;
    auto next_dt = next.get_output_layout().data_type;
    auto use_onednn_impls = _optimization_attributes.use_onednn_impls;

    if (prev.is_dynamic() || next.is_dynamic())
        return false;

    auto is_input_idx = [&](size_t idx) -> bool {
        if (&next.get_dependency(idx) == &prev)
            return true;
        if (next.get_dependency(idx).is_type<reorder>() && &next.get_dependency(idx).get_dependency(0) == &prev)
            return true;
        return false;
    };

    // Not to fuse reorder if this removal changes input format of its next node which has reuse in fused_op
    if (next.get_preferred_impl_type() == impl_types::onednn) {
        for (auto& fused_op : next.get_fused_primitives()) {
            if (fused_op.is_type<eltwise>()) {
                auto out_layout = next.get_output_layout();
                auto add_type = onednn_add_fusing_helpers::get_add_fusing_type(next, fused_op);
                if (add_type == add_fusing_type::sum && prev.get_output_layout().format != out_layout.format)
                    return false;
            }
        }
    }

    // Ref kernels are the main for depth_to_space and region_yolo and strided_slice. It can do anything.
    if (next.is_type<depth_to_space>() || next.is_type<region_yolo>() ||
        (next.is_type<strided_slice>() && next.get_preferred_impl_type() != cldnn::impl_types::cpu))
        return true;

    if (next.is_type<reorder>())
        return true;

    // Check whether the reorder between prev and next is the first input of next.
    auto is_input_reorder = [](program_node& prev, program_node& next) {
        auto found_reorder = std::find_if(next.get_dependencies().begin(), next.get_dependencies().end(), [](const std::pair<program_node*, int32_t>& dep){
            return dep.first->is_type<reorder>();
        });
        // if there is no reorder between prev and next, it returns true.
        // This case is needed for can_fuse_reorder in reorder_inputs pass.
        if (found_reorder == std::end(next.get_dependencies()) && next.get_dependency_index(prev) == 0)
            return true;
        auto& next_dep = next.get_dependency(0);
        if (!next_dep.is_type<reorder>())
            return false;
        for (auto& prev_usr : prev.get_users()) {
            if (!prev_usr->is_type<reorder>())
                continue;
            if (&next_dep == prev_usr && next.get_dependency_index(next_dep) == 0) {
                return true;
            }
        }
        return false;
    };

    // Errata for onednn layout selection. First conv can receive both bfyx and byxf directly.
    if (next.is_type<convolution>() &&
        next.get_preferred_impl_type() == impl_types::onednn &&
        ((fmt_prev == format::byxf && fmt_next == format::byxf) ||
         (fmt_prev == format::bfyx && fmt_next == format::byxf &&
            (prev_dt == data_types::f16 && next.get_input_layout(0).feature() <= 8))) &&
        is_input_reorder(prev, next))
        return true;

    // Do not remove reorder if it is necessary to fulfill required_input
    auto& reorder_node = next.get_dependency(0);
    auto reorder_layout = reorder_node.get_output_layout();
    if (reorder_layout.format == next.get_preferred_input_fmt(next.get_dependency_index(reorder_node))
            && !reorder_layout.data_padding)
        return false;

    // resample_opt kernel can work cross-layout between fsv16 and fsv32
    if (next.is_type<resample>() &&
        (fmt_prev == format::b_fs_yx_fsv16 || fmt_prev == format::b_fs_yx_fsv32
            || fmt_prev == format::bs_fs_yx_bsv16_fsv16 || fmt_prev == format::bs_fs_yx_bsv16_fsv32
            || fmt_prev == format::bs_fs_yx_bsv32_fsv16 || fmt_prev == format::bs_fs_yx_bsv32_fsv32) &&
        (fmt_next == format::b_fs_yx_fsv16 || fmt_next == format::b_fs_yx_fsv32
            || fmt_next == format::bs_fs_yx_bsv16_fsv16 || fmt_next == format::bs_fs_yx_bsv16_fsv32
            || fmt_next == format::bs_fs_yx_bsv32_fsv16 || fmt_next == format::bs_fs_yx_bsv32_fsv32))
        return true;

    if (next.is_type<pooling>() &&
        (((prev_simple && next_simple) && (prev_dt == next_dt)) ||
        ((fmt_prev == format::b_fs_yx_fsv4 && fmt_next == format::bfyx) && (prev_dt == data_types::u8 || prev_dt == data_types::i8))))
        return true;

    if (next.is_type<eltwise>() && prev_simple && next_simple)
        return true;

    if (next.is_type<fully_connected>() &&
        (fmt_prev == format::bfyx || fmt_prev == format::yxfb ||
         fmt_prev == format::b_fs_yx_fsv16 || fmt_prev == format::fs_b_yx_fsv32 ||
         fmt_prev == format::b_fs_yx_fsv32 ||
         (fmt_prev == format::b_fs_yx_fsv4 &&
          prev_output_layout.feature() % 32 == 0 &&
          prev_output_layout.spatial(0) == 1 &&
          prev_output_layout.spatial(1) == 1)) && is_input_reorder(prev, next))
        return true;

    if (next.is_type<convolution>() && fmt_prev == format::b_fs_yx_fsv16 && fmt_next == format::b_fs_yx_fsv4 && is_input_idx(0))
        return true;

    if (next.is_type<quantize>() && (fmt_prev == format::bfyx || fmt_prev == format::bfzyx) &&
        prev.is_input() && (prev_dt == data_types::u8 || prev_dt == data_types::i8))
        return true;

    if (!use_onednn_impls || next.get_preferred_impl_type() == impl_types::ocl) {
        if (next.is_type<convolution>() &&
            (fmt_prev == format::bfyx || fmt_prev == format::bs_fs_yx_bsv4_fsv2) &&
            ((fmt_next == format::fs_b_yx_fsv32 && next.as<convolution>().get_primitive()->groups == 1) ||
            (fmt_next == format::b_fs_yx_fsv32 && (prev_output_layout.feature() == 3 || prev_output_layout.feature() == 4)) ||
            (fmt_next == format::bs_fs_yx_bsv32_fsv32 && (prev_output_layout.feature() == 3 || prev_output_layout.feature() == 4)) ||
            (fmt_next == format::bs_fs_yx_bsv32_fsv16 && (prev_output_layout.feature() == 3 || prev_output_layout.feature() == 4)) ||
            (fmt_next == format::bs_fs_yx_bsv16_fsv16 && next_output_layout.feature() % 16 == 0 && prev_output_layout.feature() == 3) ||
            (fmt_next == format::bs_fs_yx_bsv16_fsv16 && next_output_layout.feature() >= 16 && prev_output_layout.feature() == 3 &&
            (next_output_layout.data_type != data_types::i8 && next_output_layout.data_type != data_types::u8))))
            return true;

        if (next.is_type<convolution>() &&
            fmt_prev == format::bfyx &&
            (fmt_next == format::b_fs_yx_fsv16 || fmt_next == format::bs_fs_yx_bsv32_fsv16) &&
            next_output_layout.feature() >= 16 && prev_output_layout.feature() <= 4 &&
            next.as<convolution>().get_primitive()->activations_zero_points.empty() &&
            next.as<convolution>().get_primitive()->weights_zero_points.empty())
            return true;

        if (next.is_type<convolution>() &&
            (fmt_prev == format::b_fs_yx_fsv4 || fmt_prev == format::bs_fs_yx_bsv4_fsv4 || fmt_prev == format::bs_fs_yx_bsv8_fsv4) &&
            ((fmt_next == format::b_fs_yx_fsv32 && (prev_output_layout.feature() == 3 || prev_output_layout.feature() == 4)) ||
            (fmt_next == format::bs_fs_yx_bsv32_fsv32 && (prev_output_layout.feature() == 3 || prev_output_layout.feature() == 4)) ||
            (fmt_next == format::bs_fs_yx_bsv4_fsv4 && (prev_output_layout.feature() == 3 || prev_output_layout.feature() == 4)) ||
            (fmt_next == format::bs_fs_yx_bsv8_fsv4 && (prev_output_layout.feature() == 3 || prev_output_layout.feature() == 4)) ||
            (fmt_next == format::b_fs_yx_fsv16 && next_output_layout.feature() >= 16 &&
            (prev_output_layout.feature() == 3 || (prev_output_layout.feature() == 4 && (prev_dt == data_types::u8 || prev_dt == data_types::i8))))))
            return true;
    }

    if (next.is_type<quantize>() && (fmt_prev == format::bfyx || fmt_prev == format::bfzyx) &&
        (fmt_next == format::b_fs_yx_fsv16 || fmt_next == format::b_fs_zyx_fsv16 ||
         fmt_next == format::bs_fs_yx_bsv16_fsv16 || fmt_next == format::b_fs_yx_fsv4))
        return true;

    if (next.is_type<convolution>() &&
        !(prev.is_type<quantize>() && (prev_dt == data_types::i8 || prev_dt == data_types::u8)) &&
        (fmt_prev == format::b_fs_yx_fsv4 || fmt_prev == format::bfyx)  && prev_output_layout.feature() == 3 &&
        (fmt_next == format::b_fs_yx_fsv4 ||
         fmt_next == format::bs_fs_yx_bsv16_fsv16))
        return true;

    if (next.is_type<convolution>() &&
        fmt_prev == format::bfyx &&
        ((fmt_next == format::b_fs_yx_fsv16 || fmt_next == format::bs_fs_yx_bsv16_fsv16) &&
            next_output_layout.feature() >= 16 && prev_output_layout.feature() == 3))
        return true;

    if (next.is_type<convolution>() &&
        fmt_prev == format::bfzyx &&
        ((fmt_next == format::b_fs_zyx_fsv16 || fmt_next == format::bs_fs_zyx_bsv16_fsv16) &&
            next_output_layout.feature() >= 16 && prev_output_layout.feature() == 3))
        return true;

    if (use_onednn_impls) {
        if (next.is_type<eltwise>() && (fmt_prev == format::bfyx) && (fmt_next == format::bs_fs_yx_bsv4_fsv2) &&
            prev.is_input() && (prev_dt == data_types::u8 || prev_dt == data_types::i8))
            return true;

        // Remove Reorder for Convolution if mixed layout.
        auto& node = prev.get_users().front();
        if (prev.get_output_layout().format == next.get_preferred_input_fmt() &&
                node->get_output_layout().data_padding == prev.get_output_layout().data_padding)
            return true;

        if (next.is_type<convolution>() &&
            (fmt_prev == format::bfyx && fmt_next == format::bs_fs_yx_bsv32_fsv32) &&
            // Condition to avoid execution in reorder_inputs.
            prev.get_users().size() == 1 && prev.get_users().front()->is_type<reorder>()) {
            const auto& cur = prev.get_users().front();
            std::set<size_t> dep_idx_set;
            for (auto& p : next.get_fused_primitives()) {
                // find eltwise sum primitive which has dependency nodes, and gather dependency indices of it.
                if (p.is_type<eltwise>() && p.typed_desc<eltwise>()->mode == eltwise_mode::sum) {
                    for (size_t i = p.outer_dep_start_idx; i < p.outer_dep_start_idx + p.total_num_deps; i++) {
                        dep_idx_set.insert(i);
                    }
                }
            }
            // The current reorder can be fused if it is a dependency of eltwise sum primitive fused.
            for (size_t i = 0; i < next.get_dependencies().size(); i++) {
                auto& d_node = next.get_dependency(i);
                if (cur->id() == d_node.id() && dep_idx_set.find(i) != dep_idx_set.end()) {
                    return true;
                }
            }
        }

        if (next.is_type<quantize>() && prev.get_users().size() == 1)
            return true;
    }

    return false;
}

bool layout_optimizer::can_fuse_reorder_to_prev(program_node& prev, reorder_node& node, format fmt_prev, format fmt_next) {
    bool allow_new_shape_infer = node.get_program().is_new_shape_infer();
    // Because mvn and concatenation kernel can work cross-layout, if reorder only performs type conversion,
    // fusing reorder to the previous node can be done even if it is a dynamic shape case
    if ((prev.is_type<mvn>() || prev.is_type<concatenation>()) &&
        !prev.is_in_shape_of_subgraph() && node.is_type_conversion_only() &&
        (format::is_simple_data_format(fmt_prev) && format::is_simple_data_format(fmt_next)))
        return true;

    if (prev.is_dynamic() || (!node.get_users().empty() && node.get_users().front()->is_dynamic()))
        return false;

    // Ref kernels are the main for depth_to_space, region_yolo and detection_output. It can do anything. Should not see next.
    if (prev.is_type<depth_to_space>() || prev.is_type<region_yolo>()
        || (prev.is_type<detection_output>() && prev.get_preferred_impl_type() != cldnn::impl_types::cpu))
        return true;

    if (node.get_users().empty())
        return false;

    auto next = node.get_users().front();
    auto dt_prev = prev.get_output_layout().data_type;
    auto dt_next = next->get_output_layout().data_type;
    auto use_onednn_impls = _optimization_attributes.use_onednn_impls;

    if (prev.is_type<reorder>())
        return true;

    // resample_opt kernel can work cross-layout between fsv16 and fsv32
    if (prev.is_type<resample>() &&
        (fmt_prev == format::b_fs_yx_fsv16 || fmt_prev == format::b_fs_yx_fsv32
            || fmt_prev == format::bs_fs_yx_bsv16_fsv16 || fmt_prev == format::bs_fs_yx_bsv16_fsv32
            || fmt_prev == format::bs_fs_yx_bsv32_fsv16 || fmt_prev == format::bs_fs_yx_bsv32_fsv32) &&
        (fmt_next == format::b_fs_yx_fsv16 || fmt_next == format::b_fs_yx_fsv32
            || fmt_next == format::bs_fs_yx_bsv16_fsv16 || fmt_next == format::bs_fs_yx_bsv16_fsv32
            || fmt_next == format::bs_fs_yx_bsv32_fsv16 || fmt_next == format::bs_fs_yx_bsv32_fsv32))
        return true;

    if (prev.is_type<one_hot>() &&
        !data_type_traits::is_floating_point(dt_prev) &&
        data_type_traits::is_floating_point(dt_next) &&
        fmt_prev == fmt_next)
        return true;

    if (prev.is_type<quantize>() &&
        (fmt_next == format::b_fs_yx_fsv4 || fmt_next == format::b_fs_zyx_fsv32 || fmt_next == format::b_fs_yx_fsv32 ||
         fmt_next == format::b_fs_yx_fsv16 || fmt_next == format::b_fs_zyx_fsv16 ||fmt_next == format::bs_fs_yx_bsv16_fsv16))
        return true;

    if (prev.is_type<permute>()) {
        if (fmt_prev == format::b_fs_yx_fsv32 && fmt_next == format::byxf)
            return true;

        auto& permute_order = prev.as<permute>().get_primitive()->permute_order;
        if ((fmt_prev == format::b_fs_yx_fsv4 || fmt_prev == format::b_fs_yx_fsv32 || fmt_prev == format::b_fs_zyx_fsv32 ||
         fmt_prev == format::b_fs_yx_fsv16 || fmt_prev == format::b_fs_zyx_fsv16 || fmt_prev == format::bs_fs_yx_bsv16_fsv16)
         && permute_order.back() != 1
         && (!prev.as<permute>().is_rotating_except_batch())) {
            return false;
        }
        // permute kernel doesn't support reorder fusion for ranks > 6
        if (fmt_prev.dimension() > 6 || fmt_next.dimension() > 6)
            return false;

        // Skip reorder fusing to permute when allow_new_shape_infer is True and input and output rank is different
        if (allow_new_shape_infer && (fmt_prev.dimension() != fmt_next.dimension()))
            return false;

        return true;
    }


    // Remove Reorder after convolution if possible.
    if (use_onednn_impls) {
        auto reorder_layout = node.get_output_layout();
        if (reorder_layout.format == prev.get_preferred_output_fmt() &&
                reorder_layout.data_padding == prev.get_output_layout().data_padding)
            return true;

        if (prev.is_type<eltwise>() &&
            is_mixed_layout(prev, *next, false, {{ format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv16 }}))
            return true;
    }

    return false;
}

namespace {
bool should_use_winograd_2x3_s1(const convolution_node& node,
                                layout const& input_layout,
                                layout const& weights_layout,
                                bool output_size_handling_enabled) {
    bool disable_winograd_conv = node.get_program().get_config().get_property(ov::intel_gpu::disable_winograd_convolution);
    if (disable_winograd_conv)
        return false;

    // cases when NOT to use winograd
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_winograd_conv == 1)
        return false;

    auto prim = node.get_primitive();
    if (input_layout.data_type != data_types::f16
        || input_layout.feature() % 64 != 0  // current algorithm is effective for ifm to be multiply of 64
        || weights_layout.spatial(0) != 3     // weights have to be 3x3 by definiton
        || weights_layout.spatial(1) != 3     // weights have to be 3x3 by definition
        || weights_layout.batch() % 64 != 0  // current algorithm is effective for ofm to be multiply of 64
        || any_not_one(prim->stride)               // stride has to be 1x1 by definition
        || any_not_one(prim->dilation)             // no support for dilation
        || output_size_handling_enabled            // This condition is weird. Need to revise it and replace with something meaningful
        || (input_layout.count() > 3000000)        // limit max input size as winograd consumes more memory
        || (input_layout.count() < 50000)          // limit min input size as winograd is not effective for small input
        || (input_layout.spatial(0) < 8 &&
            input_layout.spatial(1) < 8)      // disable winograd for small spatials as perf is poor
        || prim->groups != 1) {                    // disable winograd for groups
        return false;
    }
    return true;
}
}  // namespace

layout_optimizer::layout_optimizer(bool output_size_handling_enabled)
    : _optimization_attributes(), _output_size_handling_enabled(output_size_handling_enabled), _total_conv(0) {
    for (auto& format : optimized_formats) {
        _optimized_conv_count.insert({format, 0});
    }
}

bool layout_optimizer::is_depthwise(const convolution_node& node) const {
        const int32_t output_channels = node.get_output_layout(0).feature();
        const int32_t input_channels = node.get_input_layout(0).feature();

        return node.get_groups() == static_cast<uint32_t>(input_channels) && input_channels == output_channels;
}

bool layout_optimizer::convolution_bfyx_opt(layout const& output_layout,
                                            const layout& weights_layout,
                                            std::shared_ptr<const convolution> conv) {
    // A set of rules that define when bfyx mem format has better performance than yxfb
    if (output_layout.batch() == 16 || output_layout.batch() % 16 != 0 ||
        output_layout.data_type != data_types::f16 || weights_layout.batch() % 16 != 0 ||
        !((weights_layout.spatial(0) == 1 && weights_layout.spatial(1) == 1) ||
          (weights_layout.spatial(0) >= 5 && weights_layout.spatial(1) >= 5) ||
          (conv->stride[0] > 1 && conv->stride[1] > 1) ||
          (weights_layout.feature() <= 32 && output_layout.spatial(0) < 224 &&
           output_layout.spatial(1) < 224) ||
          (weights_layout.feature() <= 64 && output_layout.spatial(0) < 112 &&
           output_layout.spatial(1) < 112) ||
          (weights_layout.feature() <= 128 && output_layout.spatial(0) < 56 &&
           output_layout.spatial(1) < 56) ||
          (weights_layout.feature() <= 256 && output_layout.spatial(0) < 28 &&
           output_layout.spatial(1) < 28) ||
          (weights_layout.feature() <= 512 && output_layout.spatial(0) < 14 &&
           output_layout.spatial(1) < 14) ||
          (weights_layout.feature() <= 1024 && output_layout.spatial(0) <= 7 &&
           output_layout.spatial(1) <= 7)) ||
        // WA for AgeGender, which has one convolution that is better on yxfb, but due to additonal reorder overall
        // performance is worse than bfyx
        (output_layout.spatial(0) == 82 && output_layout.spatial(1) == 82) ||
        (output_layout.batch() >= 128) ||
        _optimization_attributes.bfyx_only_layer)
        return true;

    return false;
}

bool layout_optimizer::convolution_byxf_opt(const layout& input_layout,
                                            layout const& output_layout,
                                            const layout& weights_layout,
                                            const convolution_node& node) {
    auto conv = node.get_primitive();

    if (node.get_dependency(0).is_type<convolution>()) {
        auto& dep = node.get_dependency(0).as<convolution>();
        if (is_depthwise(dep))
            return false;
    }

    // A set of rules that define when byxf mem format has better performance
    if ((output_layout.data_type == data_types::f16 && weights_layout.spatial(0) == 1 &&
        all_ones(conv->dilation) &&
        !node.get_transposed() &&
         node.get_groups() == 1 &&
         input_layout.feature() % 32 == 0 &&
         weights_layout.spatial(1) == 1 && output_layout.feature() % 64 == 0 &&
         weights_layout.batch() % 64 == 0 &&
         all_ones(conv->stride) &&
         all_zeroes(conv->padding_begin) &&
         all_zeroes(conv->padding_end)) ||
        // Winograd
        should_use_winograd_2x3_s1(node, input_layout, weights_layout, _output_size_handling_enabled))
        return true;

    return false;
}

bool layout_optimizer::convolution_b_fs_yx_fsv16_opt(const layout& input_layout,
                                                     const layout& output_layout,
                                                     const layout& weights_layout,
                                                     std::shared_ptr<const convolution> conv,
                                                     bool weak_restrictions) {
    // A set of rules that define when b_fs_yx_fsv16 mem format can be used for int8 case
    bool i8_dt_case = (input_layout.data_type == data_types::u8 || input_layout.data_type == data_types::i8) &&
                       weights_layout.data_type == data_types::i8;

    if (i8_dt_case) {
        auto ks_x = weights_layout.spatial(0);
        auto ks_y = weights_layout.spatial(1);

        size_t in_features_per_group = input_layout.feature() / conv->groups;
        size_t out_features_per_group = output_layout.feature() / conv->groups;

        // Check for non-grouped or depthwise convolution
        if (input_layout.format.dimension() == 4 &&
            ((ks_x == 7 && ks_y == 7) || (ks_x == 3 && ks_y == 3) || (ks_x == 1 && ks_y == 1) || (ks_x == 5 && ks_y == 5)) &&
            output_layout.feature() >= 16 &&
            ((conv->groups == 1) ||
             conv->groups == static_cast<uint32_t>(input_layout.feature())))
            return true;
        // Check for grouped convolution
        else if (input_layout.format.dimension() == 4 && input_layout.batch() < 16 &&
                 out_features_per_group >= 16 &&
                 // Need to extend imad fsv4 kernel to handle e.g. 3 input features per group
                 (in_features_per_group % 4 == 0) &&
                 ((conv->dilation[conv->dilation.size() - 1] + 1) * (ks_x - 1)) <= 16)
                return true;
        // Check for fsv16 imad kernel
        else if ((input_layout.format.dimension() == 4) &&
                 ((in_features_per_group > 8) || (out_features_per_group >= 4)))
                return true;
        return false;
    }
    // A set of rules that define when b_fs_yx_fsv16 mem format can be used for fp16/fp32 case
    int32_t feature_block_size = 16;
    bool correct_data_type = (input_layout.data_type == data_types::f16 || input_layout.data_type == data_types::f32) &&
                             (weights_layout.data_type == input_layout.data_type);
    bool correct_batch = (input_layout.batch() == 1) || (input_layout.batch() > 1 && input_layout.data_type == data_types::f32);
    bool correct_spatial_dims = input_layout.spatial(2) == 1 && input_layout.spatial(3) == 1;
    int32_t required_feature_num = weak_restrictions ? feature_block_size / 2 : feature_block_size;
    bool correct_in_feature = (input_layout.feature() >= required_feature_num &&
                                  output_layout.feature() >= required_feature_num);
    int32_t in_features_per_group = input_layout.feature() / conv->groups;
    int32_t out_features_per_group = output_layout.feature() / conv->groups;
    if (!correct_in_feature && input_layout.feature() <= 4 && out_features_per_group >= feature_block_size)
        correct_in_feature = true;
    bool depthwise = conv->groups == static_cast<uint32_t>(input_layout.feature());  // depthwise conv
    bool grouped = ((feature_block_size % out_features_per_group == 0) &&
                       (feature_block_size % in_features_per_group == 0) &&
                       (feature_block_size / out_features_per_group > 1) &&
                       (feature_block_size / in_features_per_group > 1) &&
                       (out_features_per_group != 1) &&
                       (in_features_per_group != 1)) ||
                      ((out_features_per_group % feature_block_size == 0 || feature_block_size % out_features_per_group == 0) &&
                       (in_features_per_group % feature_block_size == 0));
    if (correct_data_type &&
        correct_batch &&
        correct_spatial_dims &&
        correct_in_feature &&
        (conv->groups == 1 || depthwise || grouped))
        return true;
    return false;
}

static bool has_reorder_before_mvn(const program_node& node, size_t cur_depth, size_t max_depth, uint64_t reorder_size_threshold = 0) {
    // MVN with rank size 3 always requires Reorder and Reshape. Due to this pattern, too many Reorder may occur when used with Convolution,
    // which may cause performance degradation. It stands out in Stable-Diffusion Unet and Decoder.
    if (cur_depth > max_depth) return false;
    if (node.is_type<reorder>()) {
        if (node.get_users().size() == 1) {
            auto reorder_first_user = node.get_users().front();
            if (reorder_first_user->is_type<reshape>()) {
                for (auto& reshape_user : reorder_first_user->get_users()) {
                    if (reshape_user->is_type<mvn>() && node.get_output_layout().get_linear_size() > reorder_size_threshold) {
                        GPU_DEBUG_LOG << node.id() << ": " << node.get_output_layout().to_short_string() << " : heavy reorder" << std::endl;
                        return true;
                    }
                }
            }
        }
    }
    bool res = false;
    for (const auto& usr : node.get_users()) {
        res |= has_reorder_before_mvn(*usr, cur_depth + 1, max_depth, reorder_size_threshold);
    }
    return res;
}

bool layout_optimizer::should_select_b_fs_yx_fsv16_layout(convolution_node const& node, layout const& weights_layout) {
    auto prim = node.get_primitive();
    auto input_layout = node.get_input_layout(0);
    auto const cond_denom = _total_conv > 0 ? 1.0f / static_cast<float>(_total_conv) : 1.0f;
    auto fully_support_conv_num = _optimized_conv_count.at({format::b_fs_yx_fsv16, false});
    auto partially_support_conv_num = _optimized_conv_count.at({format::b_fs_yx_fsv16, true});

    auto output_layout = node.calc_output_layout();

    auto current_conv_supports_layout = convolution_b_fs_yx_fsv16_opt(input_layout, output_layout, weights_layout,  prim);
    auto is_prev_conv_node_supports_layout = node.get_dependency(0).is_type<convolution>() &&
                                             is_format_optimized(node.get_dependency(0).as<convolution>(), format::b_fs_yx_fsv16);
    auto weak_restriction_cond = (partially_support_conv_num - fully_support_conv_num) * cond_denom < 0.15f;
    auto current_conv_partially_supports_layout = convolution_b_fs_yx_fsv16_opt(input_layout, output_layout, weights_layout, prim, true);
    auto may_use_weak_restrictions = is_prev_conv_node_supports_layout || weak_restriction_cond;

    return (((_optimization_attributes.b_fs_yx_fsv16_network) &&
            (current_conv_supports_layout || (may_use_weak_restrictions && current_conv_partially_supports_layout))) ||
           input_layout.format == format::b_fs_yx_fsv16) &&
           !has_reorder_before_mvn(reinterpret_cast<program_node const&>(node), 0, 3, 8300000);
            // Decoder in Stable-Diffusion showed planar format Convolution without Reorder is better than
            // blocked format Convolution in case Reorder is larger than [1, 512, 128, 128].
}

bool layout_optimizer::convolution_b_fs_zyx_fsv16_opt(const layout& input_layout,
                                                      const layout& output_layout,
                                                      const layout& weights_layout,
                                                      std::shared_ptr<const convolution> conv) {
    // A set of rules that define when b_fs_zyx_fsv16 mem format can be used
    size_t in_features_per_group = input_layout.feature() / conv->groups;
    size_t out_features_per_group = output_layout.feature() / conv->groups;

    // Check for fsv16 imad kernel
    if ((input_layout.format.dimension() == 5) &&
        (input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8) &&
        (weights_layout.data_type == data_types::i8 || weights_layout.data_type == data_types::u8) &&
        ((in_features_per_group > 8) || (out_features_per_group >= 4)))
        return true;

    bool format_ver = (input_layout.format == format::bfzyx || input_layout.format == format::b_fs_zyx_fsv16 ||
                      input_layout.format == format::bs_fs_zyx_bsv16_fsv16);
    bool data_type_ver = input_layout.data_type == data_types::f16 || input_layout.data_type == data_types::f32;
    bool w_layout = weights_layout.data_type == input_layout.data_type;
    bool single_dilation = all_ones(conv->dilation);
    bool groups_ver = conv->groups == 1 || out_features_per_group % 16 == 0
        || (conv->groups > 1 && out_features_per_group == 8);

    return format_ver && data_type_ver && w_layout && single_dilation && groups_ver;
}
bool layout_optimizer::convolution_bs_fs_yx_bsv16_fsv16_opt(const layout& input_layout,
                                                            const layout& output_layout,
                                                            const layout& weights_layout,
                                                            std::shared_ptr<const convolution> conv) {
    // A set of rules that define when bs_fs_yx_bsv16_fsv16 mem format can be used
    bool correct_batch = input_layout.batch() > 16;
    bool correct_feature = (input_layout.feature() % 16 == 0 || input_layout.feature() == 3) && output_layout.feature() % 16 == 0;
    bool fp16_ver = input_layout.data_type == data_types::f16 && input_layout.batch() % 32 == 0;
    bool fp32_ver = input_layout.data_type == data_types::f32 && input_layout.batch() % 16 == 0;
    bool single_group = conv->groups == 1;

    bool int8_sup = (input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8);
    if (int8_sup)
        correct_batch = input_layout.batch() >= 16;
    int8_sup &= (input_layout.batch() % 16 == 0 && weights_layout.data_type == data_types::i8 &&
                 conv->activations_zero_points.empty() && conv->weights_zero_points.empty());
    auto ks_x = weights_layout.spatial(0);
    auto ks_y = weights_layout.spatial(1);
    int8_sup &= (input_layout.spatial(2) == 1 && ((ks_x == 1 && ks_y == 1) || (ks_x == 3 && ks_y == 3) || (ks_x == 7 && ks_y == 7)) &&
                 output_layout.feature() % 32 == 0 && all_ones(conv->dilation));

    return (int8_sup || fp16_ver || fp32_ver) && correct_feature && correct_batch && single_group;
}

bool layout_optimizer::convolution_fs_b_yx_fsv32_opt(const layout& input_layout,
                                                     const layout& output_layout,
                                                     const layout& weights_layout,
                                                     std::shared_ptr<const convolution> conv,
                                                     bool weak_restrictions) {
    auto ofm = output_layout.feature();
    // A set of rules that define when fs_b_yx_fsv32 mem format can be used
    bool correct_batch = input_layout.batch() > 1;
    bool correct_in_feature = input_layout.feature() >= 16;
    bool correct_out_feature = weak_restrictions ? ofm >= 16 : ofm > 16;
    bool dw_conv = static_cast<int>(conv->groups) == input_layout.feature();
    if (!correct_in_feature && input_layout.feature() == 3 && conv->groups == 1) {   // bfyx with 3 feature -> fs_b_yx_fsv32 case
        correct_in_feature = true;
    }

    if (input_layout.data_type != data_types::f16 || weights_layout.data_type != data_types::f16) {
        return false;
    }

    if ((input_layout.format == format::fs_b_yx_fsv32) || (correct_out_feature && correct_in_feature && correct_batch &&
        (dw_conv || conv->groups == 1) )) {
        return true;
    }
    return false;
}

bool layout_optimizer::deconvolution_b_fs_zyx_fsv16_opt(layout const &input_layout,
                                                        const layout &/*weights_layout*/,
                                                        std::shared_ptr<const deconvolution> deconv) {
    // A set of rules that define when b_fs_zyx_fsv16 mem format can be used
    if ((input_layout.format == format::bfzyx ||
         input_layout.format == format::b_fs_zyx_fsv16 ||
         input_layout.format == format::bs_fs_zyx_bsv16_fsv16) &&
        (input_layout.data_type == data_types::f32 || input_layout.data_type == data_types::f16))
        return true;

    if (input_layout.format.dimension() == 5 &&
        (input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8))
        return true;

    return false;
}

bool layout_optimizer::deconvolution_b_fs_yx_fsv16_opt(layout const &input_layout,
                                                       const layout &weights_layout,
                                                       std::shared_ptr<const deconvolution> deconv) {
    // A set of rules that define when b_fs_yx_fsv16 mem format can be used
    if ((input_layout.format == format::bfyx || input_layout.format == format::b_fs_yx_fsv16) &&
        (input_layout.data_type == data_types::f32 || input_layout.data_type == data_types::f16) &&
        (deconv->groups == 1 || (static_cast<int>(deconv->groups) == weights_layout.group())))
        return true;

    if (input_layout.format.dimension() == 4 &&
        (input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8))
        return true;

    return false;
}

static bool is_node_for_onednn(reduce_node const& node, format preferred_format) {
    auto& input = node.input();
    auto reduce_prim = node.get_primitive();

    if (input.get_output_layout().data_type == data_types::f32
        && node.get_output_layout().data_type == data_types::f32) {
        return false;
    }

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
            if (input.get_output_layout().data_type == data_types::f16 ||
                input.get_output_layout().data_type == data_types::f32)
                break;
        default:
            return false;
    }

    auto input_layout = input.get_output_layout();

    // redundant reduce is not acceptable on oneDNN reduction
    if (node.get_output_layout() == input_layout) {
        return false;
    }

    // oneDNN reduction selects ref kernel for simple formats(bfyx..) which has perf regression with a decent tensor size.
    if (format::is_simple_data_format(preferred_format))
        return false;

    // Onednn reduction does NOT support reordering of unreduced-axes.
    // Currently, an Onednn reduce layer which contains reduction of blocked axes(b-f) is expected to select planar format.
    if (reduce_prim->keep_dims == false && is_reduce_blocked_axes(node))
        return false;

    return true;
}

static bool is_node_for_onednn(convolution_node const& node) {
    if (!layout_optimizer::are_data_types_suitable_for_onednn((program_node&)node))
        return false;

    return true;
}

static bool is_node_for_onednn(deconvolution_node const& node) {
    auto prim = node.get_primitive();
    auto input_layout = node.get_input_layout(0);
    auto output_layout = node.get_output_layout(0);

    bool onednn_valid_dt = layout_optimizer::are_data_types_suitable_for_onednn((program_node&)node);

    bool onednn_valid_params = onednn_valid_dt &&
                               prim->groups == 1 &&
                               get_post_ops_count(node) <= 32;

    auto spatial_dims_num = input_layout.get_spatial_rank();

    return onednn_valid_params && spatial_dims_num <= 3;
}


static bool is_node_for_onednn(fully_connected_node const& node) {
    auto fc_prim = node.get_primitive();

    if (fc_prim->compressed_weights) {
        auto weights_dt = node.weights().get_output_layout().data_type;
        if (!fc_prim->decompression_zero_point.empty()) {
            auto decompression_zp_idx = fc_prim->bias.empty() ? 3 : 4;
            auto decompression_zp_dt = node.get_input_layout(decompression_zp_idx).data_type;
            if (weights_dt != decompression_zp_dt)
                return false;
        }
    }

    if (!layout_optimizer::are_data_types_suitable_for_onednn((program_node&)node))
        return false;

    auto output_layout = node.get_output_layout();
    auto ps = output_layout.get_partial_shape();
    size_t non_spatial_count = 2 + (fc_prim->input_size == 3 ? 1 : 0);
    size_t rank = ps.size();

    // OneDnn doesn't support spatial dimensions for output
    for (auto i = non_spatial_count; i < rank; i++) {
        if (ps[i].is_dynamic() || ps[i] != 1) {
            return false;
        }
    }

    return true;
}

static bool is_node_for_onednn(gemm_node const& node) {
    if (!layout_optimizer::are_data_types_suitable_for_onednn((program_node&)node))
        return false;

    return true;
}

// This function is needed to avoid performance regressions for the convolutions with byxf layout
// Previously some topologies had scale operations which prevented byxf usage
// Now instead of scale we have eltwise + fused_ops which might enable byxf convolution in unexpected cases
// So here we check that given eltwise node is replacement of the old scale primitive
// TODO: Adjust byxf convolution selection logic
static bool is_scale_shift(const eltwise_node& node) {
    if (node.get_dependencies().size() != 2)
        return false;

    if (node.get_primitive()->mode != eltwise_mode::prod)
        return false;

    if (node.get_fused_primitives().empty())
        return false;

    auto fused_op0 = node.get_fused_primitives().front();

    if (!fused_op0.is_type<eltwise>())
        return false;

    if (fused_op0.typed_desc<eltwise>()->mode != eltwise_mode::sum)
        return false;

    return true;
}

bool layout_optimizer::users_for_convolution_byxf_opt(program_node const& node, uint32_t depth) {
    // This function checks if byxf optimization can be applied to the required depth of node's users.
    // Setting depth to 1 will check only node's users, depth = 2 are user's users etc.
    if (depth == 0)
        return true;

    for (auto& user : node.get_users()) {
        // primitives that support transitions byxf->other format and other format->byxf are valid for byxf opt
        if ((user->type() == cldnn::eltwise::type_id() && !is_scale_shift(user->as<eltwise>())) || user->type() == cldnn::pooling::type_id()) {
            if (!users_for_convolution_byxf_opt(*user, depth - 1))
                return false;
        // convolution that is capable to use byxf and is performant is also valid for byxf opt
        } else if (user->type() == cldnn::convolution::type_id()) {
            if (convolution_byxf_opt(node.get_output_layout(),
                                     user->calc_output_layout(),
                                     user->get_input_layout(1),
                                     user->as<convolution>())) {
                if (!users_for_convolution_byxf_opt(*user, depth - 1))
                    return false;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

bool layout_optimizer::deps_for_convolution_byxf_opt(program_node const& node, uint32_t depth) {
    // This function checks if requested format is the same for node's users in the required depth.
    // Setting depth to 1 will check only node's dependencies, depth = 2 are dep's dependencies etc.
    if (depth == 0)
        return true;

    for (auto& dep : node.get_dependencies()) {
        // skip data layers
        if (dep.first->is_type<data>())
            continue;

        if (dep.first->is_type<convolution>()) {
            auto& conv_dep = dep.first->as<convolution>();
            if (!convolution_byxf_opt(conv_dep.get_input_layout(),
                                      conv_dep.get_output_layout(),
                                      conv_dep.weights().get_output_layout(),
                                      conv_dep)) {
                return false;
            }
        } else if ((!dep.first->is_type<pooling>() && !dep.first->is_type<eltwise>()) ||
                   (dep.first->is_type<eltwise>() && is_scale_shift(dep.first->as<eltwise>()))) {
            return false;
        }

        if (!deps_for_convolution_byxf_opt(*dep.first, depth - 1))
            return false;
    }
    return true;
}

format layout_optimizer::imad_case(convolution_node const& node) const {
    auto dims_count = format::dimension(node.get_input_layout().format);

    bool is_grouped = node.get_groups() > 1;
    bool is_dw = is_depthwise(node);

    if (dims_count == 5 && is_grouped) {
        return format::bfzyx;
    } else if (dims_count == 4 && is_grouped && !is_dw) {
        return format::b_fs_yx_fsv4;
    }

    bool asymmetric_quantization = node.activations_zero_points_term() || node.weights_zero_points_term();

    if (asymmetric_quantization && _optimization_attributes.b_fs_zyx_fsv32_network) {
        if (dims_count == 5) {
            return format::b_fs_zyx_fsv32;
        } else {
            return format::b_fs_yx_fsv32;
        }
    }

    if (dims_count == 5) {
        return format::bfzyx;
    }

    return format::b_fs_yx_fsv4;
}

bool layout_optimizer::is_mixed_layout(program_node& prev, program_node& next, bool check_data_type,
                                       std::vector<std::pair<format, format>> custom_list) const {
    auto prev_layout = prev.get_output_layout();
    auto prev_fmt = prev_layout.format;
    auto prev_dt = prev_layout.data_type;
    auto next_layout = next.get_output_layout();
    auto next_fmt = next_layout.format;
    auto next_dt = next_layout.data_type;

    // std::pair<i8_u8_format, float_format>
    std::vector<std::pair<format, format>> supported_list = {
        { format::b_fs_yx_fsv32, format::b_fs_yx_fsv16 },
        { format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv32_fsv16 },
        { format::bs_fs_yx_bsv32_fsv32, format::bs_fs_yx_bsv16_fsv16 },
        { format::b_fs_zyx_fsv32, format::b_fs_zyx_fsv16 },
        { format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv32_fsv16 },
        { format::bs_fs_zyx_bsv32_fsv32, format::bs_fs_zyx_bsv16_fsv16 },
    };

    auto& check_list = custom_list.size() > 0 ? custom_list : supported_list;

    for (auto& pair : check_list) {
        if ((prev_fmt == pair.first && next_fmt == pair.second) &&
            (!check_data_type || (data_type_traits::is_i8_u8(prev_dt) && data_type_traits::is_floating_point(next_dt)))) {
            if ((next_fmt == format::bs_fs_yx_bsv32_fsv16 || next_fmt == format::bs_fs_zyx_bsv32_fsv16) && (next_dt == data_types::f32)) return false;
            if ((next_fmt == format::bs_fs_yx_bsv16_fsv16 || next_fmt == format::bs_fs_zyx_bsv16_fsv16) && (next_dt == data_types::f16)) return false;
            return true;
        }
        if ((next_fmt == pair.first && prev_fmt == pair.second) &&
            (!check_data_type || (data_type_traits::is_i8_u8(next_dt) && data_type_traits::is_floating_point(prev_dt)))) {
            if ((prev_fmt == format::bs_fs_yx_bsv32_fsv16 || prev_fmt == format::bs_fs_zyx_bsv32_fsv16) && (prev_dt == data_types::f32)) return false;
            if ((prev_fmt == format::bs_fs_yx_bsv16_fsv16 || prev_fmt == format::bs_fs_zyx_bsv16_fsv16) && (prev_dt == data_types::f16)) return false;
            return true;
        }
    }

    return false;
}

format layout_optimizer::get_expected_format(convolution_node const& node) {
    auto prim = node.get_primitive();
    auto input_layout = node.get_input_layout(0);
    auto output_layout = node.get_output_layout(0);
    auto weights_layout = node.weights().get_output_layout().convert_to_weights_layout(prim->grouped_weights_shape);
    auto expected_format = output_layout.format;
    bool i8_u8_input = input_layout.data_type == data_types::u8 || input_layout.data_type == data_types::i8;

    if (prim->deformable_mode) {
        return format::adjust_to_rank(format::bfyx, output_layout.get_partial_shape().size());
    }

    bool onednn_valid_post_ops = get_post_ops_count(node) <= 32;
    bool use_onednn_impls = _optimization_attributes.use_onednn_impls && input_layout.data_type != data_types::f32;

    // Use planar bfyx format for dynamic convolutions with explicit padding in clDNN
    if (node.is_dynamic() && output_layout.get_partial_shape().size() == 4 && node.use_explicit_padding() && !i8_u8_input &&
        !(use_onednn_impls && onednn_valid_post_ops && !node.has_padded_dependency())) {
        return format::bfyx;
    }

    if (input_layout.is_dynamic() || output_layout.is_dynamic()) {
        if (input_layout.get_partial_shape().size() <= 4)
            expected_format = format::b_fs_yx_fsv16;
        else if (input_layout.get_partial_shape().size() == 5)
            expected_format = format::b_fs_zyx_fsv16;
        return expected_format;
    }

    const float cond_denom = _total_conv > 0 ? 1.0f / static_cast<float>(_total_conv) : 1.0f;

    if (use_onednn_impls && onednn_valid_post_ops && node.get_preferred_output_fmt() != format::any) {
        expected_format = node.get_preferred_output_fmt();
    } else {
        /* *************************** Native impls format selection part ************************** */
        if (use_onednn_impls && i8_u8_input) {
            // It is here because of post operation condition for onednn.
            // Use fsv32 for onednn friendliness.
            expected_format = cldnn::format::b_fs_yx_fsv32;
        } else if (i8_u8_input) {
            if ((_optimization_attributes.b_fs_yx_fsv16_network &&
                convolution_b_fs_yx_fsv16_opt(input_layout, output_layout, weights_layout, prim))) {
                expected_format = cldnn::format::b_fs_yx_fsv16;
            } else if ((_optimization_attributes.b_fs_zyx_fsv16_network &&
                convolution_b_fs_zyx_fsv16_opt(input_layout, output_layout, weights_layout, prim))) {
                expected_format = cldnn::format::b_fs_zyx_fsv16;
            } else {
                expected_format = imad_case(node);
            }
        } else if (_optimization_attributes.b_fs_zyx_fsv16_network &&
                convolution_b_fs_zyx_fsv16_opt(input_layout, output_layout, weights_layout, prim)) {
            if ((output_layout.data_type == data_types::f32 && output_layout.batch() % 16 == 0) ||
                (output_layout.data_type == data_types::f16 && output_layout.batch() % 32 == 0))
                expected_format = cldnn::format::bs_fs_zyx_bsv16_fsv16;
            else
                expected_format = cldnn::format::b_fs_zyx_fsv16;

        } else if (output_layout.format == format::bfzyx) {
            expected_format = cldnn::format::bfzyx;
        } else if (_optimization_attributes.bs_fs_yx_bsv16_fsv16_network &&
                convolution_bs_fs_yx_bsv16_fsv16_opt(node.get_input_layout(), output_layout, weights_layout, prim)) {
            expected_format = cldnn::format::bs_fs_yx_bsv16_fsv16;
        } else if (_optimization_attributes.fs_b_yx_fsv32_network && !node.get_transposed() &&
                ((convolution_fs_b_yx_fsv32_opt(input_layout,
                                                output_layout,
                                                weights_layout, prim) ||
                (((node.get_dependency(0).is_type<convolution>() && is_format_optimized(node.get_dependency(0).as<convolution>(), format::fs_b_yx_fsv32))
                  || (_optimized_conv_count.at({format::fs_b_yx_fsv32, false}) * cond_denom > 0.8f)) &&
                  convolution_fs_b_yx_fsv32_opt(input_layout,
                                                output_layout,
                                                weights_layout, prim, true)))) &&
                 !(has_reorder_before_mvn(reinterpret_cast<program_node const&>(*node.get_users().front()), 0, 3, 1000000) &&
                     !static_cast<bool>(prepare_padding::get_needed_padding_for_convolution(const_cast<convolution_node&>(node))))) {
            // Chose fs_b_yx_fsv32 layout in two cases: 1-st: the current conv primitive totally supports fs_b_yx_fsv32 layout
            //                                          2-nd: the previous conv primitive supports fs_b_yx_fsv32 layout and
            //                                                current conv primitives supports this one with weak restrictions -
            //                                                that should be cheaper than reordering data to another layout
            expected_format = format::fs_b_yx_fsv32;
        } else if (should_select_b_fs_yx_fsv16_layout(node, weights_layout)) {
            expected_format = cldnn::format::b_fs_yx_fsv16;
        } else if (output_layout.data_type == data_types::f16 &&
                    layout_optimizer::convolution_byxf_opt(input_layout, output_layout, weights_layout, node) &&
                    (users_for_convolution_byxf_opt(node, 2) || deps_for_convolution_byxf_opt(node, 2)) &&
                    // todo: remove this condition when yxfb optimizations will be disabled
                    output_layout.format != cldnn::format::yxfb && output_layout.batch() == 1) {
            expected_format = cldnn::format::byxf;
        } else if (output_layout.format == format::b_fs_yx_fsv4 ||
                    output_layout.format == format::os_is_yx_osv16_isv4) {
            // imad case
            // nothing to do, just go out from here.
        } else if (layout_optimizer::convolution_bfyx_opt(output_layout, weights_layout, prim) || _output_size_handling_enabled || node.get_transposed()) {
            {
                if (output_layout.format == format::b_fs_zyx_fsv16 || output_layout.format == format::bs_fs_zyx_bsv16_fsv16)
                    expected_format = cldnn::format::bfzyx;
                else
                    expected_format = cldnn::format::bfyx;
            }
        } else {
            expected_format = cldnn::format::yxfb;
        }
    }

    return expected_format;
}

format layout_optimizer::get_expected_format(deconvolution_node const& node) {
    auto prim = node.get_primitive();
    auto input_layout = node.get_input_layout(0);
    auto output_layout = node.get_output_layout(0);
    auto weights_layout = node.weights().get_output_layout().convert_to_weights_layout(node.get_primitive()->grouped_weights_shape);
    auto expected_format = output_layout.format;

    if (input_layout.is_dynamic() || output_layout.is_dynamic()) {
        if (input_layout.get_partial_shape().size() <= 4)
            expected_format = format::b_fs_yx_fsv16;
        else if (input_layout.get_partial_shape().size() == 5)
            expected_format = format::b_fs_zyx_fsv16;
        return expected_format;
    }

    auto expected_shape = output_layout.get_shape();
    bool use_onednn_impls = _optimization_attributes.use_onednn_impls;

    if (use_onednn_impls && is_node_for_onednn(node)) {
        // XXX: need to take the situation into consideration where it is called from prepare_primitive_fusing
        expected_format = node.get_preferred_output_fmt();
    } else if (_optimization_attributes.b_fs_zyx_fsv16_network &&
        deconvolution_b_fs_zyx_fsv16_opt(output_layout, weights_layout, prim)) {
        if ((output_layout.data_type == data_types::f32 && expected_shape[0] % 16 == 0) ||
            (output_layout.data_type == data_types::f16 && expected_shape[0] % 32 == 0))
            expected_format = cldnn::format::bs_fs_zyx_bsv16_fsv16;
        else
            expected_format = cldnn::format::b_fs_zyx_fsv16;
    } else if ((_optimization_attributes.b_fs_yx_fsv16_network) &&
               deconvolution_b_fs_yx_fsv16_opt(output_layout, weights_layout, prim)) {
        auto input_shape = input_layout.get_shape();
        auto input_features = input_shape[1];
        auto output_features = expected_shape[1];
        float f_cost = static_cast<float>(input_features * output_features) / (align_to(input_features, 16) * align_to(output_features, 16));
        float stride_cost = 1 / static_cast<float>(prim->stride[prim->stride.size() - 1]);
        if (f_cost * stride_cost > 0.1f)
            expected_format = cldnn::format::b_fs_yx_fsv16;
        else
            expected_format = cldnn::format::bfyx;
    }
    return expected_format;
}

format layout_optimizer::get_expected_format(quantize_node const& node) {
    auto layout = node.get_output_layout();
    auto expected = format::any;

    std::function<bool(const program_node& node)> only_gemm_users = [&](const program_node& node) {
        bool all_users_gemm = true;

        for (auto user : node.get_users()) {
            if (user->is_type<reorder>() || user->is_type<reshape>())
                all_users_gemm &= only_gemm_users(*user);
            else if (user->is_type<gemm>())
                all_users_gemm &= true;
            else
                return false;
        }

        return all_users_gemm;
    };

    auto use_onednn_impls = _optimization_attributes.use_onednn_impls;

    if (use_onednn_impls) {
        auto& user = node.get_users().front();
        if (user != nullptr && user->get_preferred_input_fmt(user->get_dependency_index(node)) != format::any) {
            expected = user->get_preferred_input_fmt(user->get_dependency_index(node));
        } else {
            expected = format::any;
        }
    } else if (only_gemm_users(node)) {
        // TODO: Gemm is not supporting fsv layouts
        expected = format::get_default_format(node.get_output_layout().format.dimension());
        // TODO: check other types for first conv
    } else if (layout.is_static() && layout.format.spatial_num() == 2 &&
                (layout.data_type == data_types::i8 || layout.data_type == data_types::u8) &&
                layout.batch() % 16 == 0) {
        if (layout.feature() > 8) {
            expected = format::b_fs_yx_fsv16;
        } else {
            expected = format::b_fs_yx_fsv4;
        }
    } else if (layout.format.spatial_num() == 3 && (layout.data_type == data_types::i8 || layout.data_type == data_types::u8)) {
        expected = format::b_fs_zyx_fsv16;
    }

    // In case of input -> ... -> quantize -> concat
    if (layout.is_static() && expected == format::any
        && (node.get_users().size() == 1 && node.get_users().front()->is_type<concatenation>())
        && (layout.batch() < 4 && layout.feature() < 4)) {
        expected = format::get_default_format(layout.get_rank(), false, false);
    }

    return expected;
}

bool layout_optimizer::is_node_suitable_for_onednn(program_node& node) {
    if (node.is_type<convolution>()) {
        return is_node_for_onednn(node.as<convolution>());
    } else if (node.is_type<deconvolution>()) {
        return is_node_for_onednn(node.as<deconvolution>());
    } else if (node.is_type<fully_connected>()) {
        return is_node_for_onednn(node.as<fully_connected>());
    } else if (node.is_type<gemm>()) {
        return is_node_for_onednn(node.as<gemm>());
    }

    return false;
}

bool layout_optimizer::are_data_types_suitable_for_onednn(program_node& node) {
    auto in_dt = node.get_input_layout(0).data_type;
    auto out_dt = node.get_output_layout(false).data_type;

    // Generally, fp32 input does NOT use oneDNN
    if (in_dt == data_types::f32 &&
        (!node.is_type<fully_connected>() && !node.is_type<convolution>() && !node.is_type<reorder>()))
          return false;

    if (in_dt == data_types::i64 || out_dt == data_types::i64)
        return false;

    if (node.is_type<pooling>()) {
        return onednn_check_data_types_for_pooling(in_dt, out_dt);
    } else if (node.is_type<convolution>()) {
        auto wei_dt = node.as<convolution>().weights().get_output_layout().data_type;
        return onednn_check_data_types_for_convolution(in_dt, wei_dt, out_dt);
    } else if (node.is_type<deconvolution>()) {
        auto wei_dt = node.as<deconvolution>().weights().get_output_layout().data_type;
        return onednn_check_data_types_for_deconvolution(in_dt, wei_dt, out_dt);
    } else if (node.is_type<fully_connected>() || node.is_type<gemm>()) {
        bool is_fc = node.is_type<fully_connected>();
        data_types wei_dt;
        if (is_fc) {
            const auto& fc_node = node.as<fully_connected>();
            const auto fc_prim = fc_node.get_primitive();
            wei_dt = fc_node.weights().get_output_layout(false).data_type;
            if (fc_prim->compressed_weights)
                return true;
        } else {
            wei_dt = node.as<gemm>().get_input_layout(1).data_type;
        }
        return onednn_check_data_types_for_fc_gemm(in_dt, wei_dt, out_dt);
    } else if (node.is_type<reorder>()) {
        auto input_fmt = node.get_input_layout(0).format;
        auto output_fmt = node.get_output_layout().format;

        // For mixed precision case, oneDNN is slower than clDNN
        if (input_fmt == format::b_fs_yx_fsv16 && data_type_traits::is_i8_u8(in_dt))
            return false;
        if (output_fmt == format::b_fs_yx_fsv16 && data_type_traits::is_i8_u8(in_dt))
            return false;
        if (output_fmt == format::bfyx && out_dt == data_types::f32)
            return false;

        return true;
    }
    return false;
}

bool layout_optimizer::are_layouts_suitable_for_onednn(program_node& node) {
    auto input_layout = node.get_dependencies().front().first->get_output_layout();
    auto in_padding = input_layout.data_padding;
    auto out_padding = node.get_output_layout().data_padding;
    // Check if padding exists
    if (node.get_preferred_impl_type() == impl_types::onednn && (in_padding || out_padding)) {
        // Check spatial padding
        bool no_spatial_padding = true;
        for (size_t i = 0; i < in_padding.lower_size().spatial.size(); ++i) {
            no_spatial_padding &= (in_padding.lower_size().spatial[i] == 0);
        }
        for (size_t i = 0; i < in_padding.upper_size().spatial.size(); ++i) {
            no_spatial_padding &= (in_padding.upper_size().spatial[i] == 0);
        }
        for (size_t i = 0; i < out_padding.lower_size().spatial.size(); ++i) {
            no_spatial_padding &= (out_padding.lower_size().spatial[i] == 0);
        }
        for (size_t i = 0; i < out_padding.upper_size().spatial.size(); ++i) {
            no_spatial_padding &= (out_padding.upper_size().spatial[i] == 0);
        }

        // Onednn supports outer padding of batch axis (first element offset) if its format is 'bxxx'
        bool no_batch_padding = true;
        auto out_fmt = node.get_output_layout().format;
        if (format::is_multi_blocked(input_layout.format) || format::is_multi_blocked(out_fmt) ||
            input_layout.format.dims_order()[0] != 0 || out_fmt.dims_order()[0] != 0) {
            for (size_t i = 0; i < in_padding.lower_size().batch.size(); ++i) {
                no_batch_padding &= (in_padding.lower_size().batch[i] == 0);
            }
            for (size_t i = 0; i < in_padding.upper_size().batch.size(); ++i) {
                no_batch_padding &= (in_padding.upper_size().batch[i] == 0);
            }
            for (size_t i = 0; i < out_padding.lower_size().batch.size(); ++i) {
                no_batch_padding &= (out_padding.lower_size().batch[i] == 0);
            }
            for (size_t i = 0; i < out_padding.upper_size().batch.size(); ++i) {
                no_batch_padding &= (out_padding.upper_size().batch[i] == 0);
            }
        }
        return (no_spatial_padding && no_batch_padding);
    }
    return true;
}

bool layout_optimizer::is_primitive_implemented_for_onednn(program_node& node) {
    if (node.is_type<fully_connected>() || node.is_type<gemm>() || node.is_type<pooling>() ||
        node.is_type<convolution>() || node.is_type<deconvolution>() ||
        node.is_type<reduce>() || node.is_type<reorder>() || node.is_type<concatenation>()) {
        return true;
    }

    return false;
}

bool layout_optimizer::onednn_check_preferred_impl_type_of_users(program_node& node) {
    if (node.get_users().size() == 0)
        return false;

    for (auto& user : node.get_users()) {
        if (user->get_preferred_impl_type() == impl_types::onednn)
            return true;
    }

    return false;
}

impl_types layout_optimizer::get_forced_impl_type_by_config(program_node& node) {
#ifdef GPU_DEBUG_CONFIG
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(!debug_config->forced_impl_types.empty()) {
        // Forcing impl type of one primitive
        for (const auto& forced_impl_type : debug_config->forced_impl_types) {
            if (node.is_type<fully_connected>()) {
                if (forced_impl_type == "fc:ocl")
                    return impl_types::ocl;
                else if (forced_impl_type == "fc:onednn")
                    return impl_types::onednn;
            } else if (node.is_type<gemm>()) {
                if (forced_impl_type == "gemm:ocl")
                    return impl_types::ocl;
                else if (forced_impl_type == "gemm:onednn")
                    return impl_types::onednn;
            } else if (node.is_type<detection_output>()) {
                if (forced_impl_type == "do:cpu")
                    return impl_types::cpu;
                else if (forced_impl_type == "do:ocl")
                    return impl_types::ocl;
            } else if (node.is_type<reduce>()) {
                if (forced_impl_type == "reduce:ocl")
                    return impl_types::ocl;
                else if (forced_impl_type == "reduce:onednn")
                    return impl_types::onednn;
            } else if (node.is_type<concatenation>()) {
                if (forced_impl_type == "concat:ocl")
                    return impl_types::ocl;
                else if (forced_impl_type == "concat:onednn")
                    return impl_types::onednn;
            }

            // Forcing one layer
            size_t found_type = forced_impl_type.rfind(":");
            if (found_type != std::string::npos) {
                impl_types preferred_type = impl_types::any;
                auto impl_type = forced_impl_type.substr(found_type + 1);
                if (impl_type == "ocl")
                    preferred_type = impl_types::ocl;
                else if (impl_type == "onednn")
                    preferred_type = impl_types::onednn;
                else if (impl_type == "cpu")
                    preferred_type = impl_types::cpu;

                if (node.id() == forced_impl_type.substr(0, found_type)) {
                    GPU_DEBUG_LOG << " Forced implementation type : " << forced_impl_type.substr(0, found_type) << " : "
                                << forced_impl_type.substr(found_type + 1) << std::endl;
                    return preferred_type;
                }
            }
        }
    }
#endif

    return impl_types::any;
}

impl_types layout_optimizer::get_preferred_impl_type(program_node& node, format preferred_format) {
    impl_types preferred_impl = impl_types::any;
    auto forced_impl = get_forced_impl_type_by_config(node);
    if (forced_impl != impl_types::any)
        return forced_impl;

    if (node.is_in_shape_of_subgraph() && !node.is_type<reshape>())
        return impl_types::cpu;

    if (!_forcing_map.empty() && _forcing_map.count(node.id()) != 0) {
        preferred_impl = _forcing_map.at(node.id()).second;
    } else if (node.is_type<condition>()) {
        preferred_impl = impl_types::common;
    } else if (node.is_type<detection_output>()) {
        const auto& program = node.get_program();
        const auto& device_info = program.get_engine().get_device_info();
        const int64_t lws_max = device_info.max_work_group_size;
        auto& detection_output_node = node.as<detection_output>();
        auto confidence_layout = detection_output_node.confidence().get_output_layout();
        auto prim = detection_output_node.get_primitive();
        if (confidence_layout.is_dynamic()) {
            preferred_impl = impl_types::cpu;
        } else {
            auto batch_size_limitations = (device_info.supports_immad && device_info.execution_units_count >= 256) ? true : confidence_layout.batch() >= 4;
            auto can_use_ocl_impl = confidence_layout.batch() <= lws_max &&
                                    batch_size_limitations &&
                                    prim->confidence_threshold >= 0.1 &&
                                    prim->top_k <= 400 && prim->num_classes >= 16 &&
                                    confidence_layout.feature() > 10000;
            preferred_impl = can_use_ocl_impl ? impl_types::ocl : impl_types::cpu;
        }
    } else if (node.is_type<non_max_suppression>()) {
        const std::set<format> blocked_formats = {
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
        };
        if (blocked_formats.find(node.get_input_layout(0).format) != blocked_formats.end()) {
            preferred_impl = impl_types::ocl;
        } else {
            const auto& nms_node = node.as<non_max_suppression>();
            if (nms_node.get_primitive()->rotation != non_max_suppression::Rotation::NONE) {
                preferred_impl = impl_types::ocl;
            } else {
                const auto scores_layout = nms_node.input_scores().get_output_layout();
                if (scores_layout.is_dynamic()) {
                    preferred_impl = impl_types::cpu;
                } else {
                    const size_t kBatchNum = scores_layout.batch();
                    const size_t kClassNum = scores_layout.feature();
                    const size_t kNStreams =
                            static_cast<size_t>(node.get_program().get_config().get_property(ov::streams::num));
                    const size_t kKeyValue = kBatchNum * std::min(kClassNum, static_cast<size_t>(8)) * kNStreams;
                    preferred_impl = (kKeyValue > 64) ? impl_types::ocl : impl_types::cpu;
                }
            }
        }
    } else if (node.is_type<reorder>()) {
        if (!_optimization_attributes.use_onednn_impls)
            return impl_types::ocl;

        std::vector<format> onednn_optimized_fmt = {
            format::bfyx,
            format::b_fs_zyx_fsv16,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::bs_fs_zyx_bsv8_fsv4,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_yx_bsv16_fsv4,
            format::bs_fs_zyx_bsv16_fsv4,
            format::bs_fs_yx_bsv16_fsv2,
            format::bs_fs_zyx_bsv16_fsv2,
            format::bs_fs_zyx_bsv8_fsv2,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
            format::bs_fs_yx_bsv32_fsv32,
        };

        auto input_layout = node.get_input_layout(0);
        auto output_layout = node.get_output_layout();

        auto input_fmt = input_layout.format;
        auto output_fmt = output_layout.format;

        if (output_fmt == format::custom) {
            return impl_types::onednn;
        }

        preferred_impl = impl_types::onednn;

        if (std::find(onednn_optimized_fmt.begin(), onednn_optimized_fmt.end(), input_fmt) == onednn_optimized_fmt.end() ||
            std::find(onednn_optimized_fmt.begin(), onednn_optimized_fmt.end(), output_fmt) == onednn_optimized_fmt.end()) {
            preferred_impl = impl_types::ocl;
        }

        // onednn doesn't support paddings
        if (input_layout.data_padding || output_layout.data_padding) {
            preferred_impl = impl_types::ocl;
        }

        // Native impl works faster for this type of reorder
        if (input_fmt == format::bfyx && output_fmt == format::bfyx) {
            preferred_impl = impl_types::ocl;
        }

        // onednn reorder doesn't support different number of dimensions in input and output layouts
        if (input_fmt.dimension() != output_fmt.dimension()) {
            preferred_impl = impl_types::ocl;
        }

        if (!are_data_types_suitable_for_onednn(node)) {
            preferred_impl = impl_types::ocl;
        }
    } else if (node.is_type<reduce>()) {
        if (!_optimization_attributes.use_onednn_impls)
            return impl_types::ocl;

        if (is_node_for_onednn(node.as<reduce>(), preferred_format))
            return impl_types::onednn;
        else
            return impl_types::ocl;
    } else if (node.is_type<pooling>() || node.is_type<convolution>() || node.is_type<deconvolution>()) {
        if (!_optimization_attributes.use_onednn_impls)
            return impl_types::ocl;

        std::vector<format> onednn_optimized_formats = {
            format::byxf,
            format::bzyxf,
            format::b_fs_yx_fsv8,
            format::b_fs_zyx_fsv8,
            format::b_fs_yx_fsv16,
            format::b_fs_zyx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv32,
            format::bs_fs_yx_bsv4_fsv2,
            format::bs_fs_yx_bsv4_fsv4,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_zyx_bsv8_fsv2,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_zyx_bsv8_fsv4,
            format::bs_fs_yx_bsv16_fsv2,
            format::bs_fs_zyx_bsv16_fsv2,
            format::bs_fs_yx_bsv16_fsv4,
            format::bs_fs_zyx_bsv16_fsv4,
            format::bs_fs_yx_bsv16_fsv8,
            format::bs_fs_zyx_bsv16_fsv8,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bs_fs_zyx_bsv32_fsv32,
        };

        impl_types impl_candidate = impl_types::onednn;

        // Unexpected layout
        if (std::find(onednn_optimized_formats.begin(), onednn_optimized_formats.end(), preferred_format) == onednn_optimized_formats.end()) {
            impl_candidate = impl_types::ocl;
        }

        if (node.is_type<convolution>()) {
            if (!is_node_for_onednn(node.as<convolution>()))
                impl_candidate = impl_types::ocl;
        }

        if (node.is_type<deconvolution>()) {
            if (!is_node_for_onednn(node.as<deconvolution>()))
                impl_candidate = impl_types::ocl;
        }

        // [WA] oneDNN doesn't support > 32 post-ops. Remove once oneDNN improve post-ops for GPU.
        if (get_post_ops_count(node) > 32) {
           impl_candidate = impl_types::ocl;
        }

        if (!are_data_types_suitable_for_onednn(node)) {
            impl_candidate = impl_types::ocl;
        }

        for (auto& fo : node.get_fused_primitives()) {
            if (fo.is_type<activation>()) {
                // Some activations aren't implemented in oneDNN
                auto activation_prim = fo.typed_desc<activation>();
                if (activation_prim->activation_function == activation_func::negative ||
                    activation_prim->activation_function == activation_func::negation ||
                    activation_prim->activation_function == activation_func::sign)
                    impl_candidate = impl_types::ocl;
            }
        }

        // oneDNN doesn't support asymmetric weights quantization
        if (node.is_type<convolution>() && node.as<convolution>().weights_zero_points_term())
            impl_candidate = impl_types::ocl;

        preferred_impl = impl_candidate;
    } else if (node.is_type<concatenation>()) {
        if (!_optimization_attributes.use_onednn_impls)
            return impl_types::ocl;

        if (node.get_output_layout().data_type == data_types::i32 ||
            node.get_output_layout().data_type == data_types::f32)
            return impl_types::ocl;

        for (auto& dep : node.get_dependencies()) {
            if (dep.first->is_in_data_flow() && dep.first->get_preferred_impl_type() == impl_types::onednn) {
                return impl_types::onednn;
            }
        }
        if (format::is_blocked(node.get_output_layout().format)) {
            return impl_types::onednn;
        }
    // TODO: uncomment this code when onednn gemm implementations will have real perf improvements vs cldnn
    } else if (node.is_type<fully_connected>() || node.is_type<gemm>()) {
        if (!_optimization_attributes.use_onednn_impls)
            return impl_types::ocl;

        impl_types impl_candidate = impl_types::onednn;

        if (!are_data_types_suitable_for_onednn(node)) {
            impl_candidate = impl_types::ocl;
        }

        if (node.is_type<fully_connected>()) {
            if (!is_node_for_onednn(node.as<fully_connected>()))
                impl_candidate = impl_types::ocl;
        }

        if (node.is_type<gemm>()) {
            if (!is_node_for_onednn(node.as<gemm>()))
                impl_candidate = impl_types::ocl;
        }

        preferred_impl = impl_candidate;
    } else if (node.is_type<prior_box>()) {
        preferred_impl = impl_types::ocl;
    }

    return preferred_impl;
}

format layout_optimizer::get_preferred_format(program_node& node) {
    format expected = format::any;
    auto output_layout = node.get_output_layout();
    bool use_onednn_impls = _optimization_attributes.use_onednn_impls;

    bool allow_new_shape_infer = node.get_program().is_new_shape_infer();

    if (allow_new_shape_infer) {
        // Let reorder_input pass to check input format instead of output_format in forward investigation, vice versa
        auto out_lay_rank = node.get_output_layout(false).get_rank();
        auto has_reshape_user = [&](const program_node& node) -> bool {
            for (auto& user_node : node.get_users()) {
                if (user_node->is_type<reshape>())
                    return true;
            }
            return false;
        };

        // Return default format for output layout rank when user node is reshape
        // to add reorder in front of reshape in reorder_input stage instead of handle_reshpae stage.
        // It is only applied for the dynamic shape with static input shape
        if (!node.is_dynamic() &&  has_reshape_user(node))
            return format::get_default_format(out_lay_rank);

        if (node.is_type<shape_of>())
            return format::get_default_format(node.get_input_layout(0).get_rank());

        auto dep_size = node.get_dependencies().size();
        for (size_t i = 0; i < dep_size; i++) {
            auto in_lay_rank = node.get_input_layout(i).get_rank();
            const auto& shape_infer_deps = node.get_shape_infer_dependencies();
            if (std::find(shape_infer_deps.begin(), shape_infer_deps.end(), i) != shape_infer_deps.end()) {
                auto fmt = format::get_default_format(in_lay_rank, false, false);
                node.set_preferred_input_fmt(i, fmt);
            } else if (in_lay_rank != out_lay_rank) {
                auto fmt = get_preferred_format(node.get_dependency(i));
                // Check if selected format can be adjusted to the required input rank
                // If no, use default fotmat instead
                try {
                    // 7-dimention and 8-dimention only support plain format
                    if (in_lay_rank >= 7 || out_lay_rank >= 7) {
                        fmt = format::get_default_format(in_lay_rank);
                    } else {
                        format::adjust_to_rank(fmt, in_lay_rank);
                    }
                } catch (ov::Exception&) {
                    fmt = format::get_default_format(in_lay_rank);
                }
                node.set_preferred_input_fmt(i, fmt);
            }
        }

        // shape_infer_dep should be plain format because the memory is being read by ngraph shape infer as is
        if (node.is_shape_infer_dep()) {
            expected = format::get_default_format(output_layout.get_rank(), false, false);
            node.set_preferred_output_fmt(0, expected);
            return expected;
        }
    }
    if (!_forcing_map.empty() && _forcing_map.count(node.id()) != 0) {
        expected = _forcing_map.at(node.id()).first;
    } else if (node.is_type<convolution>()) {
        expected = get_expected_format(node.as<convolution>());
    } else if (node.is_type<quantize>()) {
        expected = get_expected_format(node.as<quantize>());
    } else if (node.is_type<reorder>() || node.is_type<input_layout>()) {
        if (node.is_type<reorder>() && node.as<reorder>().get_primitive()->has_surface_input()) {
            expected = format::nv12;
        } else {
            expected = node.get_output_layout().format;
        }
    } else if (node.is_type<reshape>()) {
        expected = format::get_default_format(node.get_output_layout().get_rank());
    } else if (node.is_type<deconvolution>()) {
        expected = get_expected_format(node.as<deconvolution>());
    } else if (node.is_type<mvn>()) {
        auto input_layout = node.get_input_layout(0);
        if (input_layout.data_type == data_types::f32 || input_layout.data_type == data_types::f16) {
            expected = format::get_default_format(input_layout.get_rank());
        }
    } else if (node.is_type<resample>()) {
        // if the resample is in the last part of the network and there are no users using blocked format,
        // it is better to reorder to bfyx before resample is done.
        // Skip all user format check when node is dynamic. It could cause endless recursive call in get_preferred_foramt()
        if (!node.is_dynamic() && all_users_simple_format_until_output(node, node, 0, 10)) {
            const auto& dim = format::dimension(node.get_output_layout().format);
            expected = format::get_default_format(dim, false, false);
        } else {
            expected = format::any;
        }
    } else if (node.is_type<permute>()) {
        if (node.get_dependencies().size() == 1 && node.get_dependencies().front().first->is_type<convolution>()) {
            auto& conv_node = node.get_dependencies().front().first->as<convolution>();
            const auto& fmt = get_preferred_format(conv_node);
            // if the preferred format of the previous conv of permute is fs_b_yx_fsv32,
            // it is better to set to b_fs_yx_fsv32 that supports tiled permute (permute_tile_8x8_4x4_fsv)
            // because fs_b_yx_fsv32 is only supported by permute_ref.
            if (node.as<permute>().is_rotating_except_batch() && fmt == format::fs_b_yx_fsv32) {
                expected = format::b_fs_yx_fsv32;
            }
        }
    } else if (node.is_type<reduce>()) {
        auto& reduce_node = node.as<reduce>();
        auto output_layout = reduce_node.get_output_layout();
        if (!use_onednn_impls && output_layout.is_dynamic()) {
            if (output_layout.format.dimension() > 4) {
                expected = format::get_default_format(output_layout.format.dimension());
            } else if (output_layout.format.dimension() == 4) {
                expected = format::any;
            }
        }
    } else if (node.is_type<arg_max_min>()) {
        // Set default format for issue 92967/98750
        // TODO: will remove when arg_max_min_ref supports blocked format
        expected = format::get_default_format(node.get_input_layouts()[0].get_rank(), false, false);
    } else if (node.is_type<fully_connected>() || node.is_type<gemm>()) {
        if (use_onednn_impls) {
            expected = node.get_preferred_output_fmt();
        }
        if (node.is_type<fully_connected>()) {
            if (allow_new_shape_infer) {
                // Plain input format is enforced because no available shape agnostic kernel supporting blocked format.
                // The condition will be relaxed once more shape agnostic kernels for other formats are enabled (e.g., fsv->bfyx FC optimized kernel(i8)))
                expected = format::get_default_format(node.get_input_layout(0).get_rank());
                node.set_preferred_input_fmt(0, expected);
            } else {
                auto& fc_node = node.as<fully_connected>();
                auto input_layout = fc_node.get_input_layout();
                if (input_layout.format.dimension() > 4) {
                    expected = format::bfyx;
                    node.set_preferred_input_fmt(0, format::bfyx);
                }
            }
        }
    } else if (node.is_type<gather>()) {
        // Gather needs the original input/output rank because
        // the parameters as indices, batch_dims and axis depend on the rank.
        node.set_preferred_input_fmt(0, format::get_default_format(node.as<gather>().get_primitive()->input_rank));
    } else if (node.is_type<loop>()) {
        expected = format::get_default_format(node.get_output_layout().get_rank());
    } else if (node.is_type<dft>()) {
        if (node.as<dft>().get_primitive()->mode == dft_mode::real &&
            node.as<dft>().get_primitive()->direction == dft_direction::forward) {
            node.set_preferred_input_fmt(0, format::get_default_format(node.get_input_layouts()[0].get_rank()));
        }
    }

    if (allow_new_shape_infer && node.get_preferred_input_fmt() != format::any) {
        if (node.get_preferred_output_fmt() != format::any)
            expected = node.get_preferred_output_fmt();
        node.set_preferred_output_fmt(0, expected);
    }
    return expected;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void layout_optimizer::select_preferred_formats_for_onednn(program_node& node, dnnl::primitive_desc prim_desc) {
    if (node.is_input() || !are_data_types_suitable_for_onednn(node)) {
        return;
    }

    node.init_preferred_fmt(node.get_dependencies().size(), node.get_users().size());
    if (node.is_type<convolution>() || node.is_type<deconvolution>()) {
        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            // Conv or deconv gets a preferred format for its data input based on source memory description
            // But an input format for fused post-ops should be same with an output format of conv/deconv
            size_t prim_input(-1);
            if (node.is_type<convolution>())
                prim_input = node.get_dependency_index(node.as<convolution>().input());
            if (node.is_type<deconvolution>())
                prim_input = node.get_dependency_index(node.as<deconvolution>().input());

            // Note: did not handle attribute properly. especially for zero-point
            cldnn::format src_fmt = format::any;
            if (idx == prim_input)
                src_fmt = onednn::find_data_format(prim_desc.src_desc());
            else  // Dep for fused post ops
                src_fmt = onednn::find_data_format(prim_desc.dst_desc());

            // WA: shallow convolution needs to set input format by bfyx.
            //     onednn recommended byxf for input format. It will insert reorder before shallow conv.
            if (node.is_type<convolution>() && node.get_input_layouts()[0].feature() == 3) {
                bool can_optimize_permute = false;
                // In permute-conv pattern, check if permute can be optimized
                // when the input memory of permute has been aligned like byxf format.
                // ex) pattern: input (bfyx) -> permute (byxf) -> oneDNN convolution
                //      input layout of permute: bfyx [b:1, f:416, y:416, x:3]
                //     output layout of permute: byxf [b:1, f:3, y:416, x:416]
                // In this case, it can be handled by changing only the shape of permute without the kernel execution.
                if (node.get_output_layout().get_rank() == 4 && node.get_dependency(0).is_type<permute>()) {
                    auto& pnode = node.get_dependency(0).as<permute>();
                    can_optimize_permute = pnode.get_users().size() == 1
                        && pnode.get_output_layout().data_type == node.get_output_layout().data_type
                        && !pnode.has_fused_primitives()
                        && !pnode.is_output() && pnode.get_input_layout(0).is_static()
                        && pnode.is_reverse_rotating_except_batch();
                }
                if (!can_optimize_permute) {
                    src_fmt = format::get_default_format(node.get_input_layouts()[0].get_rank(), false, false);
                } else {
                    // The size of dependencies and users must each be 1.
                    // In permute-conv pattern, the preferred format of permute should follow previous node.
                    node.get_dependency(0).init_preferred_fmt(1, 1);
                    node.get_dependency(0).set_preferred_input_fmt(0, format::bfyx);
                    node.get_dependency(0).can_be_optimized(true);
                }
            }

            node.set_preferred_input_fmt(idx, src_fmt);

            auto dst_fmt = onednn::find_data_format(prim_desc.dst_desc());
            // Errata: Best impl for shallow input conv with zero-point ops is ocl:xe_lp.
            if (node.is_type<convolution>() && src_fmt == format::bfyx) {
                auto& conv = node.as<convolution>();
                if (conv.get_input_layouts()[0].feature() <= 8 && conv.activations_zero_points_term() &&
                    conv.get_input_layouts()[0].data_type == data_types::u8 && conv.get_output_layout().data_type == data_types::u8) {
                    dst_fmt = format::b_fs_yx_fsv32;
                }
            }

            // In conv-permute pattern, sets the output format of conv to byxf so that permute can be optimized.
            // ex) oneDNN convolution -> (byxf) -> permute -> (bfyx) -> output
            //     output layout of convolution: byxf [b:1, f:128, y:2, x:2]
            //     output layout of permute:     bfyx [b:1, f:2, y:2, x:128]
            // In this case, it can be handled by changing only the shape of permute without the kernel execution.
            if (node.get_output_layout().get_rank() == 4
                && node.get_users().size() == 1 && node.get_users().front()->is_type<permute>()) {
                auto& pnode = node.get_users().front()->as<permute>();
                auto can_optimize_permute = pnode.get_output_layout().data_type == node.get_output_layout().data_type
                    && !pnode.has_fused_primitives()
                    && !pnode.is_output() && pnode.get_input_layout(0).is_static()
                    && pnode.is_rotating_except_batch();
                if (can_optimize_permute) {
                    dst_fmt = format::byxf;
                    pnode.init_preferred_fmt(1, 1);
                    pnode.set_preferred_input_fmt(0, cldnn::format::byxf);
                    pnode.set_preferred_output_fmt(0, cldnn::format::bfyx);
                    pnode.can_be_optimized(true);
                }
            }

            if (node.get_preferred_output_fmt() == format::any) {
                for (size_t usr = 0; usr < std::max<size_t>(1, node.get_users().size()); usr++)
                    node.set_preferred_output_fmt(usr, dst_fmt);
            }

            GPU_DEBUG_LOG << "select_preferred_formats:" << node.id() << ": " << fmt_to_str(src_fmt) << " --> " << fmt_to_str(dst_fmt)
                          << " For index : " << idx << std::endl;
        }
    } else if (node.is_type<fully_connected>() || node.is_type<gemm>()) {
        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            size_t out_rank = node.get_output_layout().get_rank();
            auto target_format = format::get_default_format(out_rank);

            node.set_preferred_input_fmt(idx, target_format);

            if (node.get_preferred_output_fmt() == format::any) {
                for (size_t usr = 0; usr < std::max<size_t>(1, node.get_users().size()); usr++) {
                    node.set_preferred_output_fmt(usr, target_format);
                }
            }
            GPU_DEBUG_LOG << "select_preferred_formats:" << node.id() << ": " << fmt_to_str(target_format) << " --> " << fmt_to_str(target_format)
                          << " For index : " << idx << std::endl;
        }
        bool disable_permute_fuse_onednn_gemm = false;
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->disable_onednn_permute_fusion == 1)
            disable_permute_fuse_onednn_gemm = true;
        // Optimized out permute from permute-gemm pattern. i.e. permute -> gemm
        if (node.is_type<gemm>() && !disable_permute_fuse_onednn_gemm && node.get_program().get_config().get_property(ov::intel_gpu::optimize_data)) {
            // Only the formats below support permute opt out in gemm and permute pattern. For other formats, need to check the gemm performance.
            for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
                if (node.get_dependency(idx).is_type<permute>()) {
                    auto& pnode = node.get_dependency(idx);
                    if (pnode.has_fused_primitives()) {
                        continue;
                    }
                    auto input_lay = pnode.get_dependency(0).get_output_layout();
                    auto output_lay = pnode.get_output_layout();
                    bool can_fuse_permute = input_lay.compatible(output_lay) ||
                                            ((input_lay.is_dynamic() || output_lay.is_dynamic()) &&
                                             format::is_default_format(input_lay.format) &&
                                             format::is_default_format(output_lay.format) && pnode.get_users().size() == 1);
                    const auto& permute_order = pnode.get_kernel_impl_params()->typed_desc<permute>()->permute_order;
                    std::vector<size_t> order(std::begin(permute_order), std::end(permute_order));
                    format fmt = format::bfyx;
                    if (can_fuse_permute && gemm_inst::is_fusable_permute_input_order_onednn(order, fmt)) {
                        pnode.init_preferred_fmt(1, 1);
                        pnode.set_preferred_output_fmt(0, format(static_cast<format::type>(fmt)));
                        pnode.can_be_optimized(true);
                        node.set_preferred_input_fmt(idx, format(static_cast<format::type>(fmt)));
                        GPU_DEBUG_TRACE_DETAIL << pnode.id() << " is fused to onednn gemm user : " << node.id() << std::endl;
                        GPU_DEBUG_TRACE_DETAIL << "    permute order : ";
                        GPU_DEBUG_CODE(for (const auto& o : permute_order) GPU_DEBUG_TRACE_DETAIL << o << " "; GPU_DEBUG_TRACE_DETAIL << std::endl;)
                    }
               }
            }
            // gemm -> permute
            if (node.get_users().size() == 1 && node.get_users().front()->is_type<permute>() && !node.has_fused_primitives()) {
                auto& pnode = node.get_users().front()->as<permute>();
                if (!pnode.has_fused_primitives()) {
                    auto input_lay = pnode.get_dependency(0).get_output_layout();
                    auto output_lay = pnode.get_output_layout();
                    bool can_fuse_permute = input_lay.compatible(output_lay) ||
                                            ((input_lay.is_dynamic() || output_lay.is_dynamic()) &&
                                             format::is_default_format(input_lay.format) &&
                                             format::is_default_format(output_lay.format) && pnode.get_users().size() == 1);
                    format fmt = format::bfyx;
                    auto impl_param = pnode.get_kernel_impl_params();
                    auto desc = impl_param->typed_desc<permute>();
                    auto permute_order = desc->permute_order;
                    std::vector<size_t> order(std::begin(permute_order), std::end(permute_order));
                    if (can_fuse_permute && gemm_inst::is_fusable_permute_output_order_onednn(order, fmt)) {
                        node.set_preferred_output_fmt(0, format(static_cast<format::type>(fmt)));
                        pnode.init_preferred_fmt(1, 1);
                        pnode.set_preferred_input_fmt(0, format(static_cast<format::type>(fmt)));
                        // tmp :: to fix
                        format out_fmt = format::bfyx;
                        pnode.set_preferred_output_fmt(0, format(static_cast<format::type>(out_fmt)));
                        pnode.can_be_optimized(true);
                        GPU_DEBUG_TRACE_DETAIL << pnode.id() << " is fused to onednn gemm pred : " << node.id() << std::endl;
                        GPU_DEBUG_TRACE_DETAIL << "    permute order : ";
                        GPU_DEBUG_CODE(for (const auto& o : permute_order) GPU_DEBUG_TRACE_DETAIL << o << " "; GPU_DEBUG_TRACE_DETAIL << std::endl;)
                    }
                }
            }
        }
    }
}
#endif  // ENABLE_ONEDNN_FOR_GPU

bool layout_optimizer::all_users_simple_format_until_output(program_node& origin_node, program_node& cur_node, int32_t cur_depth, int32_t max_depth) {
    if (cur_node.is_output()) return true;
    if (cur_depth > max_depth) return false;

    if (cur_node.is_type<permute>()) {
        if (!cur_node.as<permute>().is_rotating_except_batch())
            return false;
    }

    if (cur_node.is_in_data_flow() && (cur_node.type() != origin_node.type())) {
        const auto& fmt = get_preferred_format(cur_node);
        if (fmt != format::any && !format::is_simple_data_format(fmt)) {
            return false;
        }
    }

    bool res = true;
    for (const auto& usr : cur_node.get_users()) {
        res &= all_users_simple_format_until_output(origin_node, *usr, cur_depth + 1, max_depth);
    }
    return res;
}

void layout_optimizer::set_optimization_attribute(optimization_attributes_type attribute, int32_t val) {
    switch (attribute) {
        case optimization_attributes_type::group_convolution:
            _optimization_attributes.group_convolution = val;
            break;
        case optimization_attributes_type::bfyx_only_layer:
            _optimization_attributes.bfyx_only_layer = val;
            break;
        case optimization_attributes_type::fs_b_yx_fsv32_network:
            _optimization_attributes.fs_b_yx_fsv32_network = val;
            break;
        case optimization_attributes_type::b_fs_zyx_fsv32_network:
            _optimization_attributes.b_fs_zyx_fsv32_network = val;
            break;
        case optimization_attributes_type::b_fs_yx_fsv16_network:
            _optimization_attributes.b_fs_yx_fsv16_network = val;
            break;
        case optimization_attributes_type::b_fs_zyx_fsv16_network:
            _optimization_attributes.b_fs_zyx_fsv16_network = val;
            break;
        case optimization_attributes_type::bs_fs_yx_bsv16_fsv16_network:
            _optimization_attributes.bs_fs_yx_bsv16_fsv16_network = val;
            break;
        case optimization_attributes_type::use_onednn_impls:
            _optimization_attributes.use_onednn_impls = val;
            break;
        default:
            throw std::out_of_range("unsupported layout optimization attribute");
    }
}

bool layout_optimizer::is_format_optimized(const convolution_node& node, const format& format, bool use_weak_restrictions) {
    auto input_layout = node.get_input_layout();
    auto weights_layout = node.weights().get_output_layout();
    auto output_layout = node.calc_output_layout();
    auto prim = node.get_primitive();

    if (input_layout.is_dynamic() || output_layout.is_dynamic())
        return true;
    switch (format) {
        case format::b_fs_yx_fsv16:
            return convolution_b_fs_yx_fsv16_opt(input_layout, output_layout, weights_layout, prim, use_weak_restrictions) &&
                   // Work-around for inability to use b_fs_yx_fsv16 and winograd together
                   !should_use_winograd_2x3_s1(node, input_layout, weights_layout, _output_size_handling_enabled);
        case format::b_fs_zyx_fsv16:
        case format::bs_fs_zyx_bsv16_fsv16:
            return convolution_b_fs_zyx_fsv16_opt(input_layout, output_layout, weights_layout, prim);
        case format::fs_b_yx_fsv32:
            return convolution_fs_b_yx_fsv32_opt(input_layout, output_layout, weights_layout, prim);
        case format::bs_fs_yx_bsv16_fsv16:
            return convolution_bs_fs_yx_bsv16_fsv16_opt(input_layout, output_layout, weights_layout, prim);
        case format::bs_fs_yx_bsv32_fsv32:
        case format::bs_fs_yx_bsv32_fsv16:
            return false;
        default:
            throw std::invalid_argument(
                "[Layout optimizer] Other formats in is_format_optimized(...) method are not implemented!");
    }
}

void layout_optimizer::update_formats_map(const convolution_node &node) {
    for (auto& format : optimized_formats) {
        if (is_format_optimized(node, format.first, format.second)) {
            _optimized_conv_count.at(format)++;
        }
    }
    _total_conv++;
}

size_t layout_optimizer::get_total_conv_count() {
    return _total_conv;
}

size_t layout_optimizer::get_optimized_conv_count(const std::pair<format::type, bool>& format) {
    if (_optimized_conv_count.count(format) > 0) {
        return _optimized_conv_count.at(format);
    }
    return 0;
}

bool layout_optimizer::is_format_optimized(const deconvolution_node& node, const format& format) {
    auto input_layout = node.get_input_layout();
    auto weights_layout = node.weights().get_output_layout();
    auto prim = node.get_primitive();

    switch (format) {
    case format::b_fs_zyx_fsv16:
    case format::bs_fs_zyx_bsv16_fsv16:
        return deconvolution_b_fs_zyx_fsv16_opt(input_layout, weights_layout, prim);
    case format::b_fs_yx_fsv16:
        return deconvolution_b_fs_yx_fsv16_opt(input_layout, weights_layout, prim);
    default:
        throw std::invalid_argument(
            "[Layout optimizer] Other formats in is_format_optimized(...) method are not implemented!");
    }
}

void layout_optimizer::set_implementation_forcing(const ov::intel_gpu::ImplForcingMap& map) {
    for (const auto& kv : map) {
        _forcing_map.emplace(kv.first, std::make_pair(kv.second.output_format, kv.second.impl_type));
    }
}

const std::map<primitive_id, std::pair<format::type, impl_types>> layout_optimizer::get_implementation_forcing() const {
    return _forcing_map;
}

const std::vector<std::pair<format::type, bool>> layout_optimizer::optimized_formats = {
        {format::b_fs_yx_fsv16, true},
        {format::b_fs_yx_fsv16, false},
        {format::b_fs_zyx_fsv16, false},
        {format::bs_fs_zyx_bsv16_fsv16, false},
        {format::bs_fs_yx_bsv16_fsv16, false},
        {format::fs_b_yx_fsv32, false}
};
