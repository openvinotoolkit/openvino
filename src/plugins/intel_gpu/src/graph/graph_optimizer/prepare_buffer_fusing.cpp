// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "prepare_buffer_fusing.h"
#include "pooling_inst.h"
#include "kv_cache_inst.h"
#include "gather_inst.h"
#include "primitive_inst.h"
#include "activation_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "gemm_inst.h"
#include "read_value_inst.h"
#include "reshape_inst.h"
#include "permute_inst.h"
#include "depth_to_space_inst.h"
#include "resample_inst.h"
#include "loop_inst.h"
#include "lstm_elt_inst.h"
#include "strided_slice_inst.h"
#include "shape_of_inst.h"
#include "non_max_suppression_inst.h"
#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "border_inst.h"

#include "pass_manager.h"
#include "program_helpers.h"

#include <utility>
#include <list>
#include <vector>

using namespace cldnn;

namespace cldnn {
bool concat_noop_optimization::match(concatenation_node& node) {
    if (node.is_output())
        return false;
    if (node.is_dynamic())
        return false;
    return node.get_dependencies().size() == 1 && !node.has_fused_primitives();
}

bool concat_noop_optimization::optimize(concatenation_node& node) {
    auto& dep = node.get_dependency(0);
    auto outputPadding = node.get_output_layout().data_padding;
    dep.merge_output_padding(outputPadding);
    prog.extract_and_remove(node);
    // Node has been removed, so no further optimizations.
    return true;
}

bool concat_in_place_optimization::match(concatenation_node& node) {
    std::vector<kernel_impl_params> pred_params;
    for (auto pred : node.get_dependencies()) {
        pred_params.push_back(*pred.first->get_kernel_impl_params());
    }
    return (match(node, *node.get_kernel_impl_params(), pred_params));
}

// reverted condition - if any of this node's inputs is used by more than one primitive
// and is not optimized concatenation then do not fuse buffers
// TODO: we need add padding support for all optimized kernels to remove this condition
auto available_pred = [](const program_node& input) {
    if (!input.is_type<pooling>() && !input.is_type<convolution>() && !input.is_type<quantize>() &&
        !input.is_type<activation>() && !input.is_type<deconvolution>() && !input.is_type<concatenation>() &&
        !input.is_type<crop>() && !input.is_type<eltwise>() && !input.is_type<resample>() &&
        !input.is_type<reorder>() && !(input.is_type<permute>() && !input.as<permute>().is_rotating_except_batch()) &&
        !input.is_type<strided_slice>())
        return false;
    return true;
};

bool concat_in_place_optimization::match(const program_node& concat_node,
                                         kernel_impl_params& concat_params,
                                         std::vector<kernel_impl_params>& pred_params,
                                         bool is_runtime) {
    if (concat_node.is_output() || concat_params.fused_desc.size() > 0 || concat_node.is_in_shape_of_subgraph())
        return false;
    bool do_runtime_buffer_fusing = true;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_runtime_buffer_fusing) {
        do_runtime_buffer_fusing = false;
    }

    auto concat_axis = concat_params.typed_desc<concatenation>()->axis;
    size_t concat_axis_index = concat_axis < 0 ? concat_axis + concat_params.get_output_layout().get_rank() : concat_axis;
    auto def_fmt = format::get_default_format(concat_params.get_output_layout().get_rank());
    // If static padding exists in non dyn_pad axis, returns false to avoid optimized out.
    if (concat_node.is_dynamic()) {
        for (size_t j = 0; j < concat_params.get_output_layout().get_rank(); j++) {
            if (j != concat_axis_index) {
                if ((concat_params.get_output_layout().data_padding._lower_size[j] != 0)
                    || (concat_params.get_output_layout().data_padding._upper_size[j] != 0))
                    return false;
            }
        }
    }

    const auto& pred_nodes = concat_node.get_dependencies();
    for (const auto& p : pred_nodes) {
        // TODO : In dynamic shape only one user is allowed for optimzied concat
        // It is mainly because of the limited flexibility of current exec order
        // For now, we are doing shape_infer for all pred nodes and concats when executing one of the predecessors for runtime buffer fusing
        // So we need to ensure that shape_infer of the all the parents of other predecessors are done.
        // We need to shuffle the exec order for that requirement, but currently only simple method is applied which is only applicable
        // for simple patterns where the concat is the only user of all the preds.
        // Also cascaded concat is not handled for dynamic shape. for now.
        // If we have more flexible exec order handling in the future we'll be able to remove this condition below
        if (p.first->is_dynamic() && (!do_runtime_buffer_fusing || p.first->get_users().size() > 1))
            return false;
        if (concat_node.is_dynamic() && (!do_runtime_buffer_fusing || !p.first->is_dynamic()))
            return false;
    }
    // if this is called in primitive_inst::execute() and concat is static, that concat should already be optimized in build time, not in runtime.
    if (is_runtime && !concat_node.is_dynamic())
        return false;
    bool is_onednn_impl = false;

    // For in place concatenation input layouts and data types must match.
    // Also, it checks whether data along f-axis is aligned properly for implicit concat.
    // Otherwise, use explicit concat instead.
    const auto& output_format = concat_params.get_output_layout().format;
    const auto& output_datatype = concat_params.get_output_layout().data_type;

    auto lower_padd_in_axis = concat_params.get_output_layout().data_padding._lower_size[concat_axis];
    lower_padd_in_axis = std::max(lower_padd_in_axis,
                                  pred_params[0].get_output_layout().data_padding._lower_size[concat_axis]);

    size_t idx = 0;
    for (const auto& pred : pred_nodes) {
        if (!available_pred(*pred.first))
            return false;
        if (pred.first->is_output())
            return false;
        // if an input is marked as network output, prevent optimizations
        // which would affect a form of its output (unless debug flag is set),
        // we also need to restrict input types to those which support padding on all axis
        if (!pred.first->is_dynamic() || is_runtime) {
            if (!pred.first->is_padding_supported(static_cast<int>(concat_axis), lower_padd_in_axis))
                return false;
        }
        // TODO: handle optimized reshape
        if (pred.first->is_type<reshape>() && pred.first->can_be_optimized())
            return false;
        // TODO: Investigate if this condition is needed
        if (pred.first->get_users().size() > 2)
            return false;

       // Check that input isn't optimized out concatenation along different axis.
        if (pred.first->is_type<concatenation>() && pred.first->can_be_optimized()) {
            // cascaded concat opt is not supported for dynamic shape yet
            if (concat_node.is_dynamic() || is_runtime)
                return false;
            else if (pred.first->as<concatenation>().get_primitive()->axis != concat_axis)
                return false;
        }
        // Check that input isn't optimized out non-concatenation.
        if (!pred.first->is_type<concatenation>() && pred.first->can_be_optimized())
            return false;

        size_t concat_users = 0;
        for (const auto& user : pred.first->get_users())
            if (user->is_type<concatenation>())
                concat_users += 1;

        // If input is used by more than one concatenation then they may require different paddings.
        if (concat_users != 1)
            return false;

        const layout& pred_l = pred_params[idx].get_output_layout();
        if (output_format != pred_l.format || output_datatype != pred_l.data_type)
            return false;
        if (pred_l.format.block_sizes().size() > 1)
            return false;
        // TODO: Below condition should be moved to program_node::supports_padding.
        // This however will need updating the algorithm as it may make cascade adjustment impossible in some cases.
        // It however would make normal optimizations possible in others, so this is a trade-off to be investigated.
        if ((!concat_node.is_dynamic() || is_runtime) && (idx != concat_node.get_dependencies().size() - 1)) {
            if ((pred_l.format == format::b_fs_yx_fsv16 || pred_l.format == format::b_fs_zyx_fsv16) &&
                (pred_l.feature() % 16 != 0 || concat_axis != 1))
                return false;

            if ((pred_l.format == format::b_fs_yx_fsv32 || pred_l.format == format::b_fs_zyx_fsv32) &&
                (pred_l.feature() % 32 != 0 || concat_axis != 1))
                return false;

            if (pred_l.format == format::b_fs_yx_fsv4 && (pred_l.feature() != 4 || concat_axis != 1))
                return false;
        }
        if (pred.first->get_preferred_impl_type() == impl_types::onednn) {
            for (const auto& fused_op : pred_params[idx].fused_desc) {
                auto add_type = onednn_add_fusing_helpers::get_add_fusing_type(*pred.first, fused_op);
                if (add_type == add_fusing_type::sum)
                    return false;
                else
                    continue;
            }

            // Optimized-out input node is no longer onednn impl.
            if (!pred.first->can_be_optimized())
                is_onednn_impl = true;
        }
        // If sibling is using onednn impl and batch > 1, the onednn impl cannot process the implicit concat'ed buffer.
        // Onednn impls can process implicit concat'ed buffer only through buffer pointer manipulation.
        if ((!concat_node.is_dynamic() || is_runtime) && ((concat_params.get_output_layout().batch() > 1) ||
            (!concat_node.is_dynamic() && concat_params.get_output_layout().batch() > 1))) {
            for (auto& sib : pred.first->get_users()) {
                if (sib->get_preferred_impl_type() == impl_types::onednn) {
                    return false;
                }
            }
        }
        const auto& input_padd = pred.first->get_output_layout().data_padding;

        // Check that there isn't already some padding between inputs in concat axis.
        // If node has already been optimized we skip this check - this is just cascade adjustment.
        if (!concat_node.can_be_optimized()) {
            if (idx != concat_node.get_dependencies().size() && input_padd._upper_size[concat_axis] != 0)
                return false;
            if (idx != 0 && input_padd._lower_size[concat_axis] != 0)
                return false;
        }
        if (!concat_node.is_dynamic() || is_runtime)
            lower_padd_in_axis += pred_params[idx].get_output_layout().get_tensor().sizes(def_fmt)[concat_axis];
        idx++;
    }

    // Implicit concat for onednn only when use_usm and batch 1.
    if (is_onednn_impl) {
        bool use_usm = concat_node.get_program().get_engine().use_unified_shared_memory();
        const layout& concat_out_l = concat_params.get_output_layout();
        if (!use_usm)
            return false;
        if (concat_node.is_dynamic() && !is_runtime) {
            // Return true in build time, it will be checked again in runtime
            return true;
        } else {
            if (concat_out_l.batch() > 1)
                return false;
            // TODO: cldnn cases should be updated. This logic is working for onednn only.
            //       white list for support fusing formats.
            const std::vector<format> white_list = {
                format::bfyx,
                format::bfzyx,
                format::b_fs_yx_fsv16,
                format::b_fs_zyx_fsv16,
                format::b_fs_yx_fsv32,
                format::b_fs_zyx_fsv32,
                format::b_fs_yx_fsv4,
            };
            if (std::find_if(white_list.begin(), white_list.end(), [&concat_out_l](format fmt){ return (fmt == concat_out_l.format); }) == std::end(white_list))
                return false;
        }
    }
    return true;
}

void concat_in_place_optimization::optimize_cascade(concatenation_node& node, std::list<concatenation_node*>& need_reoptimization) {
     std::vector<layout> preds_layouts;
    for (auto dep : node.get_dependencies()) {
        if (dep.first->is_type<concatenation>() && dep.first->can_be_optimized())
            need_reoptimization.push_back(&dep.first->as<concatenation>());
        preds_layouts.push_back(dep.first->get_output_layout());
    }
    layout concat_layout = node.get_output_layout();
    update_in_place_concat_paddings(concat_layout, preds_layouts, node.get_primitive()->axis, false);
    size_t i = 0;
    for (auto& dep : node.get_dependencies()) {
        dep.first->set_output_layout(preds_layouts[i]);
        dep.first->can_share_buffer(false);
        ++i;
    }
    node.set_output_layout(concat_layout);
    node.can_be_optimized(true);
    if (node.is_dynamic()) {
        node.set_runtime_skippable(true);
    }
    GPU_DEBUG_TRACE_DETAIL << "[prepare_buffer_fusing] : " << node.id() << " can be optimized" << std::endl;
}

void concat_in_place_optimization::update_in_place_concat_paddings(
                                                    layout& concat_out_layout,
                                                    std::vector<layout>& preds_layouts,
                                                    size_t concat_axis,
                                                    bool is_runtime) {
    if (concat_out_layout.is_dynamic() && !is_runtime) {
        // set dynamic pad dims for shape agnostic kernel
        for (auto& dep_output_layout : preds_layouts) {
            padding::DynamicDimsMask info_dynamic_pad;
            info_dynamic_pad[concat_axis] = 1;
            dep_output_layout.data_padding._dynamic_dims_mask = info_dynamic_pad;
        }
        return;
    }

    // Select output padding by propagating all required input paddings.
    auto padd = concat_out_layout.data_padding;
    for (auto input : preds_layouts) {
        auto inputPadding = input.data_padding;
        padd = padding::max(padd, inputPadding);
    }

    std::vector<tensor::value_type> lower_padd, upper_padd;
    for (size_t i = 0; i < concat_out_layout.get_rank(); i++) {
        lower_padd.push_back(padd._lower_size[i]);
        upper_padd.push_back(padd._upper_size[i]);
    }

    // For cascade adjustment override padding in concat axis to output padding.
    // In other case match(...) already checked that only first/last input have lower/upper padding.
    lower_padd[concat_axis] = concat_out_layout.data_padding._lower_size[concat_axis];
    upper_padd[concat_axis] = concat_out_layout.data_padding._upper_size[concat_axis];
    padding::DynamicDimsMask dyn_pad_dims;
    dyn_pad_dims[concat_axis] = 1;
    concat_out_layout.data_padding = padding(lower_padd, upper_padd);

    upper_padd[concat_axis] += concat_out_layout.get_dims()[concat_axis];

     // apply concatenation in place optimization
    for (auto& pred_layout : preds_layouts) {
        auto input_length = pred_layout.get_dims()[concat_axis];
        // shrink upper pad so it points at the end of the input's buffer
        //
        //   |--- lower padd ---|                    |---------- upper padd -----------|
        //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
        upper_padd[concat_axis] -= input_length;

        // set new padding for input
        if (is_runtime)
            pred_layout.data_padding = padding(lower_padd, upper_padd, dyn_pad_dims);
        else
            pred_layout.data_padding = padding(lower_padd, upper_padd);
        // move lower padd further
        //
        //   |-------------- lower padd -------------|---------- upper padd -----------|
        //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
        lower_padd[concat_axis] += input_length;
    }
}
}  // namespace cldnn

static bool can_reshape_be_optimized(const reshape_node& node) {
    // In case if pad is not propagated, the primitive can't be optimized out
    if (!node.is_runtime_propagatable_padding()
        && node.get_input_layout(0).data_padding.is_dynamic()
        && !node.get_output_layout(0).data_padding.is_dynamic()) {
        return false;
    }

    if (node.has_fused_primitives())
        return false;

    // Onednn supports padded input of outer axis
    if (!node.is_dynamic() && node.has_outer_padding_offset() &&
        node.get_users().front()->get_preferred_impl_type() == impl_types::onednn)
        return true;

    if (node.is_in_place())
        return true;

    return false;
}

static bool is_optimizable_padding_for_crop(const crop_node& node,
                                            const layout& crop_layout,
                                            const layout& input_layout,
                                            const tensor offsets) {
    if (input_layout.data_padding._lower_size[0] != 0 || input_layout.data_padding._upper_size[0] != 0 ||
        input_layout.data_padding._lower_size[2] != 0 || input_layout.data_padding._upper_size[2] != 0 ||
        input_layout.data_padding._lower_size[3] != 0 || input_layout.data_padding._upper_size[3] != 0)
        return false;

    auto opt_lower_pad = offsets.feature[0];
    auto opt_upper_pad = input_layout.feature() - offsets.feature[0] - crop_layout.get_tensor().feature[0];

    // do not optimize crop if paddings are not properly aligned
    for (auto& usr : node.get_users()) {
        auto usr_layout = usr->get_output_layout();
        if (usr_layout.format == format::b_fs_yx_fsv16 &&
            (opt_lower_pad % 16 != 0 || opt_upper_pad % 16 != 0))
            return false;

        // oneDNN doesn't support paddings
        if (usr->get_preferred_impl_type() == impl_types::onednn)
            return false;
    }

    return true;
}

bool crop_in_place_optimization::can_crop_be_optimized_along_feature(const layout& crop_layout,
                                                                     const layout& input_layout) {
    auto format = crop_layout.format;
    const auto& crop_size = crop_layout.get_tensor();
    const auto& out_pad = crop_layout.data_padding;

    if (format == format::bfyx && crop_size.batch[0] == input_layout.batch() &&
        crop_size.spatial[0] == input_layout.spatial(0) &&
        crop_size.spatial[1] == input_layout.spatial(1) && out_pad._lower_size[1] == 0 &&
        out_pad._upper_size[1] == 0 && out_pad._lower_size[0] == 0 &&
        out_pad._upper_size[0] == 0 && out_pad._lower_size[2] == 0 &&
        out_pad._lower_size[3] == 0 && out_pad._upper_size[2] == 0 &&
        out_pad._upper_size[3] == 0) {
        return true;
    }

    return false;
}

bool crop_in_place_optimization::can_crop_be_optimized_simple_data_format(const layout& crop_layout,
                                                                          const layout& input_layout) {
    auto format = crop_layout.format;
    const auto& in_padding = input_layout.data_padding;
    const auto& out_padding = crop_layout.data_padding;

    if (format::is_simple_data_format(format) && !out_padding && !in_padding) {
        return true;
    }

    return false;
}

static bool can_read_value_be_optimize(const read_value_node& node) {
    std::unordered_set<const cldnn::program_node*> unique_users(node.get_users().begin(), node.get_users().end());
    if (unique_users.size() == 1)
        return true;

    const auto non_shape_of_users_count = std::count_if(unique_users.begin(), unique_users.end(), [](const program_node* user) {
        return !user->is_type<shape_of>();
    });
    if (non_shape_of_users_count <= 1)
        return true;

    return false;
}

static void propagate_padding_to_opt_out_users(program_node& node, cldnn::padding padding_data) {
    if (padding_data == cldnn::padding())
        return;

    for (auto user : node.get_users()) {
        if (user->can_be_optimized()) {
            user->merge_output_padding(padding_data);
            propagate_padding_to_opt_out_users(*user, padding_data);
        }
    }
}

bool crop_in_place_optimization::match(crop_node& node) {
    auto pred_param = node.get_dependency(0).get_kernel_impl_params();
    auto pred_layout = pred_param->get_output_layout();
    return (match(node, *node.get_kernel_impl_params(), pred_layout));
}

bool crop_in_place_optimization::match(const program_node& node,
                                       kernel_impl_params& crop_params,
                                       layout& input_layout,
                                       bool is_runtime) {
    if (!node.is_valid_output_layout())
        return false;
    // if the node is marked as network output, prevent optimizations which would affect a form of its output,
    // unless debug flag is set
    if (node.is_output() || crop_params.has_fused_primitives() || node.is_in_shape_of_subgraph())
        return false;

    const auto& crop_layout = crop_params.get_output_layout();
    for (auto user : node.get_users()) {
        // If the user node's output shape is already static, the padding
        // w/ dyn pad mask will not be propagated properly at runtime
        if (node.is_dynamic() && !user->get_output_pshape().is_dynamic())
            return false;
        // do not optimize when next node is concatenation which is not output
        if (user->is_type<concatenation>() && !user->is_output())
            return false;
        if (user->is_type<loop>() || user->is_type<non_max_suppression>())
            return false;
        // If the input tensor of convolution includes dynamic padding, there is an issue
        // where the total size of tensor is not properly calculated and becomes 0
        // It causes issue for internal buffer allocation during runtime
        // TODO: Need to allow optimization for gemm user
        if (node.is_dynamic() && (user->is_type<convolution>() || user->is_type<gemm>()))
            return false;
        // For static shape, gemm ref kernel is selected if there is padding on the feature, x, or y axes.
        // In such cases, do not optimize out this crop to use the opt kernel.
        // TODO: Modify gemm_tiled_opt kernel to support padding even in static shape.
        if ((!node.is_dynamic() || is_runtime) && user->is_type<gemm>() &&
            (user->get_dependency_index(node) == 0 || user->get_dependency_index(node) == 1)) {
            if (crop_params.input_offsets[0].feature[0] != 0 ||
                crop_params.input_offsets[0].spatial[0] != 0 ||
                crop_params.input_offsets[0].spatial[1] != 0) {
                return false;
            }
        }
        if (user->is_type<reshape>()) {
            // runtime buffer fusing is only handled when there is only one reshape user
            if (node.is_dynamic() && node.get_users().size() != 1)
                return false;
            auto& reshape_node = user->as<reshape>();
            if (can_reshape_be_optimized(reshape_node) &&
                (!node.is_dynamic() || !reshape_node.is_runtime_propagatable_padding()))
                return false;
        }
        if (user->is_type<experimental_detectron_roi_feature_extractor>() && user->get_dependency_index(node) == 0)
            return false;
    }

    // do not optimize crop, that must be calculated in propagate_constants
    if (node.is_constant())
        return false;

    // do not optimize variadic_split crop when either input1 or input2 is not constant.
    // VariadicSplit ngraph shape infer requires value of axis(input1) and split_lengths(input2).
    // And non_constant input1/input2 makes risky execution of runtime buffer fusing.
    auto& crop_node = node.as<crop>();
    if ((crop_node.get_primitive()->op_mode == cldnn::crop_ngraph_op_mode::variadic_split) &&
        (!crop_node.get_dependency(1).is_constant() || !crop_node.get_dependency(2).is_constant()))
        return false;

    if (node.get_users().size() > 0) {
        if (node.get_program().is_body_program() && node.get_dependency(0).is_type<lstm_elt>()) {
            return false;
        }

        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->disable_runtime_buffer_fusing && node.is_dynamic()) {
            return false;
        }

        // optimization is available for cropping across depth(features) or batch
        // if output padding has defined padding across features already it wouldn't
        // work because it expect to have zeros in the padded area.
        if ((!node.is_dynamic() || is_runtime) &&
            !is_optimizable_padding_for_crop(node, crop_layout, input_layout, crop_params.input_offsets[0]))
            return false;
        if (!(((!node.is_dynamic() || is_runtime) && can_crop_be_optimized_along_feature(crop_layout, input_layout))
            || can_crop_be_optimized_simple_data_format(crop_layout, input_layout)))
            return false;
    } else {
        return false;
    }
    return true;
}

bool crop_in_place_optimization::optimize(crop_node& node) {
    auto crop_layout = node.get_output_layout();
    auto input_layout = node.get_input_layout(0);
    auto crop_params = node.get_kernel_impl_params();

    if (crop_params->has_fused_primitives())
        return false;

    //  Regular crop
    //  crop input buffer
    //  |___________data____________|
    //
    //  crop output buffer
    //  |-------->| offsets[f]  |<--|
    //            |_____data____|
    //             <------------>
    //           reference size
    //
    //  In-place crop
    //  crop output buffer
    //  |_low_pad_|__data_size__|___|<-upper pad
    if (!node.is_dynamic() && can_crop_be_optimized_along_feature(crop_layout, input_layout)) {
        update_in_place_crop_padding_along_feature(node,
                                                   crop_layout,
                                                   input_layout,
                                                   crop_params->input_offsets[0],
                                                   node.get_primitive()->axis,
                                                   false);
    } else if (can_crop_be_optimized_simple_data_format(crop_layout, input_layout)) {
        std::pair<const program_node*, layout> user_info;
        if (node.get_users().front()->is_type<reshape>()) {
            auto& reshape_node = node.get_users().front()->as<reshape>();
            if (reshape_node.is_runtime_propagatable_padding()) {
                user_info.first = &reshape_node;
                user_info.second = reshape_node.get_output_layout();
            }
        }
        update_in_place_crop_padding_simple_data_format(crop_layout,
                                                        input_layout,
                                                        user_info,
                                                        crop_params->input_offsets[0],
                                                        node.get_primitive()->axis,
                                                        false);
        if (user_info.first) {
            node.get_users().front()->set_output_layout(user_info.second);
        }
    }
    node.set_output_layout(crop_layout);
    node.can_be_optimized(true);
    propagate_padding_to_opt_out_users(node, node.get_output_layout().data_padding);
    GPU_DEBUG_TRACE_DETAIL << "[prepare_buffer_fusing] : " << node.id() << " can be optimized" << std::endl;
    return false;
}

void crop_in_place_optimization::update_in_place_crop_padding_along_feature(const program_node& node,
                                                                            layout& crop_layout,
                                                                            layout& input_layout,
                                                                            const tensor offsets,
                                                                            size_t crop_axis,
                                                                            bool is_runtime) {
    // If it's build-time and node is dynamic, only dynamic padding is set first
    if ((crop_layout.is_dynamic() || input_layout.is_dynamic()) && !is_runtime) {
        padding::DynamicDimsMask info_dynamic_pad;
        info_dynamic_pad[crop_axis] = 1;
        crop_layout.data_padding._dynamic_dims_mask = info_dynamic_pad;
        return;
    }

    const auto& crop_size = crop_layout.get_tensor();
    const auto& out_pad = crop_layout.data_padding;

    auto opt_lower_pad = offsets.feature[0];
    auto opt_upper_pad = input_layout.feature() - offsets.feature[0] - crop_size.feature[0];

    auto& dep = node.get_dependency(0);
    //  feature num of pad should be accumulated if dep has been optimized out.
    if (dep.is_type<crop>() && dep.can_be_optimized()) {
        auto dep_pad = dep.get_output_layout().data_padding;
        opt_lower_pad += dep_pad._lower_size[1];
        opt_upper_pad += dep_pad._upper_size[1];
    }
    std::vector<int32_t> lower_sizes;
    lower_sizes.push_back(out_pad._lower_size[0]);
    lower_sizes.push_back(opt_lower_pad);
    lower_sizes.push_back(out_pad._lower_size[2]);
    lower_sizes.push_back(out_pad._lower_size[3]);
    std::vector<int32_t> upper_sizes;
    upper_sizes.push_back(out_pad._upper_size[0]);
    upper_sizes.push_back(opt_upper_pad);
    upper_sizes.push_back(out_pad._upper_size[2]);
    upper_sizes.push_back(out_pad._upper_size[3]);

    // set padding
    if (is_runtime) {
        padding::DynamicDimsMask dyn_pad_sizes;
        dyn_pad_sizes[crop_axis] = 1;
        crop_layout.data_padding = padding(lower_sizes, upper_sizes, dyn_pad_sizes);
    } else {
        crop_layout.data_padding = padding(lower_sizes, upper_sizes);
    }
}

void crop_in_place_optimization::update_in_place_crop_padding_simple_data_format(layout& crop_layout,
                                                                                 layout& input_layout,
                                                                                 std::pair<const program_node*, layout>& user_info,
                                                                                 const tensor offsets,
                                                                                 size_t crop_axis,
                                                                                 bool is_runtime) {
    // If it's build-time and node is dynamic, only dynamic padding is set first
    if ((crop_layout.is_dynamic() || input_layout.is_dynamic()) && !is_runtime) {
        padding::DynamicDimsMask dyn_pad_sizes;
        dyn_pad_sizes[crop_axis] = 1;
        crop_layout.data_padding._dynamic_dims_mask = dyn_pad_sizes;

        if (user_info.first && user_info.first->is_type<reshape>()) {
            auto reshape_desc = user_info.first->as<reshape>().get_primitive();
            auto reshape_mode = reshape_desc->mode;
            auto reshape_axis = crop_axis;
            if (reshape_mode == reshape::reshape_mode::base) {
                auto reshape_ps = user_info.second.get_partial_shape();
                auto crop_dim_val = crop_layout.get_partial_shape()[crop_axis].get_length();

                auto mul = 1;
                reshape_axis = reshape_ps.size() - 1;
                for (size_t i = reshape_ps.size(); i > 1; i--) {
                    if (reshape_ps[i - 1].is_dynamic() || mul == crop_dim_val)
                        break;

                    mul *= reshape_ps[i - 1].get_length();
                    reshape_axis = i - 1;
                }
            } else if (reshape_mode == reshape::reshape_mode::unsqueeze || reshape_mode == reshape::reshape_mode::squeeze) {
                auto reshape_ps = user_info.second.get_partial_shape();
                auto output_pattern = reshape_desc->output_pattern;

                for (size_t i = 0; i < output_pattern.size(); i++) {
                    if (output_pattern[i] <= static_cast<int64_t>(reshape_axis)) {
                        reshape_axis += reshape_mode == reshape::reshape_mode::unsqueeze ? 1 : -1;
                    }
                }
            }

            auto reshape_dyn_pad_mask = padding::DynamicDimsMask();
            reshape_dyn_pad_mask[reshape_axis] = 1;
            user_info.second.data_padding._dynamic_dims_mask = reshape_dyn_pad_mask;
        }
        return;
    }

    const auto& crop_size = crop_layout.get_tensor();

    std::vector<int32_t> lower_sizes;
    lower_sizes.push_back(offsets.batch[0]);
    lower_sizes.push_back(offsets.feature[0]);
    for (int32_t i = static_cast<int32_t>(input_layout.get_spatial_rank() - 1); i >= 0; i--) {
        lower_sizes.push_back(offsets.spatial[i]);
    }
    std::vector<int32_t> upper_sizes;
    upper_sizes.push_back(input_layout.batch() - offsets.batch[0] - crop_size.batch[0]);
    upper_sizes.push_back(input_layout.feature() - offsets.feature[0] - crop_size.feature[0]);
    for (int32_t i = static_cast<int32_t>(input_layout.get_spatial_rank() - 1); i >= 0; i--) {
        upper_sizes.push_back(input_layout.spatial(i) - offsets.spatial[i] - crop_size.spatial[i]);
    }

    if (is_runtime) {
        padding::DynamicDimsMask dyn_pad_sizes;
        dyn_pad_sizes[crop_axis] = 1;
        crop_layout.data_padding = padding(lower_sizes, upper_sizes, dyn_pad_sizes);
        if (user_info.first) {
            auto reshape_desc = user_info.first->as<reshape>().get_primitive();
            auto reshape_mode = reshape_desc->mode;
            if (reshape_mode == reshape::reshape_mode::base) {
                auto reshape_ps = user_info.second.get_partial_shape();
                auto crop_dim_val = crop_layout.get_partial_shape()[crop_axis].get_length();

                auto divider = 1;
                auto reshape_axis = reshape_ps.size();
                for (size_t i = reshape_ps.size(); i > 1; i--) {
                    const auto& dim_value = reshape_ps[i - 1].get_length();
                    if (divider * dim_value == crop_dim_val)
                        break;

                    divider *= dim_value;
                    reshape_axis = i - 1;
                }
                reshape_axis -= 1;

                const auto output_rank = std::max(reshape_ps.size(), static_cast<size_t>(4));
                std::vector<int32_t> reshape_lower_sizes(output_rank, 0);
                std::vector<int32_t> reshape_upper_sizes(output_rank, 0);
                padding::DynamicDimsMask reshape_dyn_pad_mask;

                reshape_lower_sizes[reshape_axis] = lower_sizes[crop_axis];
                reshape_upper_sizes[reshape_axis] = upper_sizes[crop_axis];
                reshape_dyn_pad_mask[reshape_axis] = 1;

                if (reshape_lower_sizes[reshape_axis])
                    reshape_lower_sizes[reshape_axis] /= divider;
                if (reshape_upper_sizes[reshape_axis])
                    reshape_upper_sizes[reshape_axis] /= divider;

                user_info.second.data_padding = padding(reshape_lower_sizes, reshape_upper_sizes, reshape_dyn_pad_mask);
            } else {
                auto reshape_ps = user_info.second.get_partial_shape();
                auto output_pattern = reshape_desc->output_pattern;

                auto reshape_axis = crop_axis;
                for (size_t i = 0; i < output_pattern.size(); i++) {
                    if (output_pattern[i] <= static_cast<int64_t>(reshape_axis)) {
                        reshape_axis += reshape_mode == reshape::reshape_mode::unsqueeze ? 1 : -1;
                    }
                }

                const auto output_rank = std::max(reshape_ps.size(), static_cast<size_t>(4));
                std::vector<int32_t> reshape_lower_sizes(output_rank, 0);
                std::vector<int32_t> reshape_upper_sizes(output_rank, 0);
                padding::DynamicDimsMask reshape_dyn_pad_mask;

                reshape_lower_sizes[reshape_axis] = lower_sizes[crop_axis];
                reshape_upper_sizes[reshape_axis] = upper_sizes[crop_axis];
                reshape_dyn_pad_mask[reshape_axis] = 1;

                user_info.second.data_padding = padding(reshape_lower_sizes, reshape_upper_sizes, reshape_dyn_pad_mask);
            }
        }
    } else {
        crop_layout.data_padding = padding(lower_sizes, upper_sizes);
    }
}

// ToDo remove friendship relation from  program_node
void prepare_buffer_fusing::run(program& p) {
    /*
    We need to take care of proper ordering by types.
    1. Concats
    2. Crops
    3. Others
    Concat before crops is needed because of the crop fusing padding requirments.
    If crop is before concat there can be padding mismatch, since concat changes padding.
    */
    auto can_optimize = [](const program_node* node) {
        bool is_dynamic = node->is_dynamic();
        bool is_planar = format::is_default_format(node->get_output_layout().format);
        bool no_pad = !node->get_output_layout().data_padding && !node->get_input_layouts().empty() && !node->get_input_layout(0).data_padding;
        if (node->is_type<read_value>() || node->is_type<kv_cache>())
            return true;

        if ((node->is_type<crop>() || node->is_type<reshape>()) && is_dynamic && is_planar && no_pad && !node->is_output() && !node->has_fused_primitives()) {
            return true;
        }

        if (is_dynamic || node->is_output() || node->has_fused_primitives() || node->is_in_shape_of_subgraph()) {
            return false;
        }
        return true;
    };

    // [1] First try to optimize all concats
    run_node_optimizations<concat_noop_optimization,
                           concat_in_place_optimization>(p);

    // [2] Then try to optimize all crops
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = (*node_itr++);
        if (!node->is_valid_output_layout())
            continue;
        if (!can_optimize(node))
            continue;

        program_helpers::do_for_types<crop>(*node, [](crop_node& node) {
            auto pred_param = node.get_dependency(0).get_kernel_impl_params();
            auto pred_layout = pred_param->get_output_layout();
            if (!crop_in_place_optimization::match(node, *node.get_kernel_impl_params(), pred_layout))
                return;

            auto crop_layout = node.get_output_layout();
            auto crop_params = node.get_kernel_impl_params();
            if (!node.is_dynamic() && crop_in_place_optimization::can_crop_be_optimized_along_feature(crop_layout, pred_layout)) {
                crop_in_place_optimization::update_in_place_crop_padding_along_feature(node,
                                                                                       crop_layout,
                                                                                       pred_layout,
                                                                                       crop_params->input_offsets[0],
                                                                                       node.get_primitive()->axis,
                                                                                       false);
            } else if (crop_in_place_optimization::can_crop_be_optimized_simple_data_format(crop_layout, pred_layout)) {
                std::pair<const program_node*, layout> user_info;
                std::vector<layout> reshape_layouts;
                if (node.get_users().front()->is_type<reshape>()) {
                    auto& reshape_node = node.get_users().front()->as<reshape>();
                    if (reshape_node.is_runtime_propagatable_padding()) {
                        user_info.first = &reshape_node;
                        user_info.second = reshape_node.get_output_layout();
                    }
                }
                crop_in_place_optimization::update_in_place_crop_padding_simple_data_format(crop_layout,
                                                                                            pred_layout,
                                                                                            user_info,
                                                                                            crop_params->input_offsets[0],
                                                                                            node.get_primitive()->axis,
                                                                                            false);
                if (user_info.first) {
                    node.get_users().front()->set_output_layout(user_info.second);
                }
            }
            node.set_output_layout(crop_layout);
            node.can_be_optimized(true);
            propagate_padding_to_opt_out_users(node, node.get_output_layout().data_padding);
            GPU_DEBUG_TRACE_DETAIL << "[prepare_buffer_fusing] : " << node.id() << " can be optimized" << std::endl;
        });
    }

    // [3] Optimize all other primitives
    node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = (*node_itr++);
        if (!node->is_valid_output_layout())
            continue;

        if (!can_optimize(node))
            continue;

        program_helpers::do_for_types<reshape>(*node, [](reshape_node& node) {
            node.get_output_layout();

            // Optimizing at prepare_buffer_fusing could propagate a padded input of an input nodes to Reshape.
            // Reshape can be optimized out when only an outer axis(batch) has padding.
            // For this case , it should re-calculate output padding size.
            if (node.has_outer_padding_offset())
                node.adjust_output_padding();

            node.can_be_optimized(can_reshape_be_optimized(node));
            GPU_DEBUG_TRACE_DETAIL << "[prepare_buffer_fusing] : " << node.id() << " can be optimized" << std::endl;
        });
        program_helpers::do_for_types<kv_cache>(*node, [](kv_cache_node& node) {
            auto kv_out_layout = node.get_output_layout();

            program_node* rv_prim = nullptr;
            program_node* gather_prim = nullptr;
            if (node.get_dependency(0).is_type<read_value>()) {
                rv_prim = &node.get_dependency(0);
            } else {
                if (node.get_dependency(0).is_type<gather>()) {
                    gather_prim = &node.get_dependency(0);
                } else {
                    return;
                }

                if (gather_prim->get_dependency(0).is_type<read_value>()) {
                    rv_prim = &gather_prim->get_dependency(0);
                }
            }

            if (!rv_prim)
                return;

            if (kv_out_layout.data_type != rv_prim->get_output_layout().data_type)
                return;

            auto concat_axis = node.get_primitive()->concat_axis;

            if (kv_out_layout.is_dynamic()) {
                // set dynamic pad dims for shape agnostic kernel
                padding::DynamicDimsMask info_dynamic_pad;
                info_dynamic_pad[concat_axis] = 1;
                kv_out_layout.data_padding._dynamic_dims_mask = info_dynamic_pad;
                node.set_output_layout(kv_out_layout);
                node.can_share_buffer(false);

                auto update_dep = [](program_node* dep, padding::DynamicDimsMask& info_dynamic_pad, size_t idx) {
                    auto prev_layout = dep->get_output_layout(true, idx);
                    prev_layout.data_padding._dynamic_dims_mask = info_dynamic_pad;
                    dep->set_output_layout(prev_layout, true, idx);
                    dep->can_share_buffer(false);
                };

                auto update_scale_zp = [&](size_t kv_cache_output_idx, size_t read_value_output_idx) {
                    auto scales_out_layout = node.get_output_layout(false, kv_cache_output_idx);

                    const size_t scales_zp_concat_axis = 2;
                    padding::DynamicDimsMask info_dynamic_pad_scales;
                    info_dynamic_pad_scales[scales_zp_concat_axis] = 1;
                    scales_out_layout.data_padding._dynamic_dims_mask = info_dynamic_pad_scales;
                    node.set_output_layout(scales_out_layout, true, kv_cache_output_idx);

                    update_dep(rv_prim, info_dynamic_pad_scales, read_value_output_idx);
                };

                if (rv_prim) {
                    update_dep(rv_prim, info_dynamic_pad, 0);
                }
                if (gather_prim) {
                    update_dep(gather_prim, info_dynamic_pad, 0);
                }

                const auto& desc = node.get_primitive();
                if (desc->compressed) {
                    update_scale_zp(2, 1);

                    if (desc->get_compression_zp_inputs_num() > 0) {
                        update_scale_zp(3, 2);
                    }
                }
            }
        });
        program_helpers::do_for_types<read_value>(*node, [](read_value_node& node) {
            // Current implementation allows to avoid copy on read_value primitive
            // only in cases when it has single user
            // Otherwise we may face an issue with exeuction of read_value users and assign to the same variable
            // Graph below is an example of unsupported case
            //     ┌────────┐     ┌───────┐
            //     │ Param1 │     │ Const │
            //     └───┬────┘     └───┬───┘
            //         │              │
            //         │         ┌────┴──────┐
            //  .......│.........│ ReadValue │
            //  .      │         └────┬─────┬┘
            //  .      │              │     │
            //  .      │   ┌─────┐    │     │
            //  .      └───┤ Add ├────┘     │
            //  .          └──┬──┘          │
            //  .             │             │
            //  .             │             │
            //  . ┌────────┐  │    ┌─────┐  │
            //  ..│ Assign ├──┴────┤ Add ├──┘
            //    └────────┘       └──┬──┘
            //                        │
            //                        │
            //                   ┌────┴──────┐
            //                   │  Result   │
            //                   └───────────┘
            // If read_value here returns variable memory w/o copy, then based on Add-s and Assign execution order we may have different results
            // TODO: Allow optimizations for the case above too. Looks like it can be achieved by more careful
            // topological sort (i.e. if we ensure that all read_value users are completed before assign is run)
            node.can_be_optimized(can_read_value_be_optimize(node));
            GPU_DEBUG_TRACE_DETAIL << "[prepare_buffer_fusing] : " << node.id() << " can be optimized = " << node.can_be_optimized() << std::endl;
        });
    }
}
