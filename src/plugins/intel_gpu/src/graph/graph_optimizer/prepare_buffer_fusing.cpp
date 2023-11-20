// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "prepare_buffer_fusing.h"
#include "intel_gpu/primitives/read_value.hpp"
#include "pooling_inst.h"
#include "primitive_inst.h"
#include "activation_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "read_value_inst.h"
#include "reshape_inst.h"
#include "depth_to_space_inst.h"
#include "resample_inst.h"
#include "loop_inst.h"
#include "strided_slice_inst.h"
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
                                         kernel_impl_params concat_params,
                                         std::vector<kernel_impl_params> pred_params,
                                         bool is_runtime) {
    if (concat_node.is_output() || concat_params.fused_desc.size() > 0 || concat_node.is_in_shape_of_subgraph())
        return false;
    bool do_runtime_buffer_fusing = true;
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->disable_runtime_buffer_fusing) {
        do_runtime_buffer_fusing = false;
    }
    auto pred_nodes = concat_node.get_dependencies();
    for (auto p : pred_nodes) {
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
    auto output_format = concat_params.get_output_layout().format;
    auto output_datatype = concat_params.get_output_layout().data_type;
    auto concat_axis = concat_params.typed_desc<concatenation>()->axis;

    auto def_fmt = format::get_default_format(concat_params.get_output_layout().get_rank());
    auto lower_padd_in_axis = concat_params.get_output_layout().data_padding.lower_size().sizes(def_fmt)[concat_axis];
    lower_padd_in_axis = std::max(lower_padd_in_axis,
                                  pred_params[0].get_output_layout().data_padding.lower_size().sizes(def_fmt)[concat_axis]);

    size_t idx = 0;
    for (auto pred : pred_nodes) {
        if (!available_pred(*pred.first))
            return false;
        if (pred.first->is_output())
            return false;
        // if an input is marked as network output, prevent optimizations
        // which would affect a form of its output (unless debug flag is set),
        // we also need to restrict input types to those which support padding on all axis
        if (!pred.first->is_dynamic() || is_runtime) {
            if (!pred.first->is_padding_supported(concat_axis, lower_padd_in_axis))
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
        for (auto& user : pred.first->get_users())
            if (user->is_type<concatenation>())
                concat_users += 1;

        // If input is used by more than one concatenation then they may require different paddings.
        if (concat_users != 1)
            return false;

        layout pred_l = pred_params[idx].get_output_layout();
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
        auto input_padd = pred.first->get_output_layout().data_padding;

        // Check that there isn't already some padding between inputs in concat axis.
        // If node has already been optimized we skip this check - this is just cascade adjustment.
        if (!concat_node.can_be_optimized()) {
            if (idx != concat_node.get_dependencies().size() && input_padd.upper_size().sizes(def_fmt)[concat_axis] != 0)
                return false;
            if (idx != 0 && input_padd.lower_size().sizes(def_fmt)[concat_axis] != 0)
                return false;
        }
        if (!concat_node.is_dynamic() || is_runtime)
            lower_padd_in_axis += pred_params[idx].get_output_layout().get_tensor().sizes(def_fmt)[concat_axis];
        idx++;
    }

    // Implicit concat for onednn only when use_usm and batch 1.
    if (is_onednn_impl) {
        bool use_usm = concat_node.get_program().get_engine().use_unified_shared_memory();
        layout concat_out_l = concat_params.get_output_layout();
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
}

void concat_in_place_optimization::update_in_place_concat_paddings(
                                                    layout& concat_out_layout,
                                                    std::vector<layout>& preds_layouts,
                                                    size_t concat_axis,
                                                    bool is_runtime) {
    auto concat_out_rank = concat_out_layout.get_rank();
    // We need to transform axis from bf[v][u][w][z]yx order to bfxy[z][w][u][v] due to tensor.sizes() usages here
    // should be removed once pad representation is changed
    auto concat_axis_legacy = concat_axis;
    if (concat_axis_legacy >= 2) {
        auto spatial_axis = concat_axis_legacy - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max<size_t>(concat_out_rank, 4) - 2;
        concat_axis_legacy = spatial_size - spatial_axis - 1 + 2;
    }

    if (concat_out_layout.is_dynamic() && !is_runtime) {
        // set dynamic pad dims for shape agnostic kernel
        for (auto& dep_output_layout : preds_layouts) {
            auto info_dynamic_pad = tensor(0).sizes();
            info_dynamic_pad[concat_axis_legacy] = 1;
            dep_output_layout.data_padding.set_dynamic_pad(tensor(info_dynamic_pad));
        }
        return;
    }

    // Select output padding by propagating all required input paddings.
    auto padd = concat_out_layout.data_padding;
    for (auto input : preds_layouts) {
        auto inputPadding = input.data_padding;
        padd = padding::max(padd, inputPadding);
    }

    auto lower_padd = padd.lower_size().sizes();
    auto upper_padd = padd.upper_size().sizes();

    // For cascade adjustment override padding in concat axis to output padding.
    // In other case match(...) already checked that only first/last input have lower/upper padding.
    lower_padd[concat_axis_legacy] = concat_out_layout.data_padding.lower_size().sizes()[concat_axis_legacy];
    upper_padd[concat_axis_legacy] = concat_out_layout.data_padding.upper_size().sizes()[concat_axis_legacy];
    auto dyn_pad_dims = lower_padd;
    dyn_pad_dims[concat_axis_legacy] = 1;
    concat_out_layout.data_padding = padding(lower_padd, upper_padd);

    upper_padd[concat_axis_legacy] += concat_out_layout.get_dims()[concat_axis];

     // apply concatenation in place optimization
    for (auto& pred_layout : preds_layouts) {
        auto input_length = pred_layout.get_dims()[concat_axis];
        // shrink upper pad so it points at the end of the input's buffer
        //
        //   |--- lower padd ---|                    |---------- upper padd -----------|
        //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
        upper_padd[concat_axis_legacy] -= input_length;

        // set new padding for input
        if (is_runtime)
            pred_layout.data_padding = padding(lower_padd, upper_padd, 0.f, tensor(dyn_pad_dims));
        else
            pred_layout.data_padding = padding(lower_padd, upper_padd, 0.f);
        // move lower padd further
        //
        //   |-------------- lower padd -------------|---------- upper padd -----------|
        //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
        lower_padd[concat_axis_legacy] += input_length;
    }
}
}  // namespace cldnn

static bool can_reshape_be_optimized(const reshape_node& node) {
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

static bool is_optimizable_padding_for_crop(const crop_node& node) {
    const auto& crop_layout = node.get_output_layout();
    auto input_layout = node.get_dependency(0).get_output_layout();
    auto crop_prim = node.get_primitive();
    auto opt_lower_pad = crop_prim->offsets.feature[0];
    auto opt_upper_pad = input_layout.feature() - crop_prim->offsets.feature[0] - crop_layout.get_tensor().feature[0];

    // do not optimize crop if paddings are not properly aligned
    for (auto& usr : node.get_users()) {
        auto usr_layout = usr->get_output_layout();
        if (usr_layout.format == format::b_fs_yx_fsv16 &&
            (opt_lower_pad % 16 != 0 || opt_upper_pad % 16 != 0))
            return false;

        if (input_layout.data_padding.lower_size().batch[0] != 0 || input_layout.data_padding.upper_size().batch[0] != 0 ||
            input_layout.data_padding.lower_size().spatial[0] != 0 || input_layout.data_padding.upper_size().spatial[0] != 0 ||
            input_layout.data_padding.lower_size().spatial[1] != 0 || input_layout.data_padding.upper_size().spatial[1] != 0)
            return false;

        // oneDNN doesn't support paddings
        if (usr->get_preferred_impl_type() == impl_types::onednn)
            return false;
    }

    return true;
}

static bool can_crop_be_optimized_along_feature(const crop_node& node) {
    const auto& crop_layout = node.get_output_layout();
    auto format = crop_layout.format;
    auto input_layout = node.get_dependency(0).get_output_layout();
    const auto& crop_size = crop_layout.get_tensor();
    const auto& out_pad = crop_layout.data_padding;

    if (format == format::bfyx && crop_size.batch[0] == input_layout.batch() &&
        crop_size.spatial[0] == input_layout.spatial(0) &&
        crop_size.spatial[1] == input_layout.spatial(1) && out_pad.lower_size().feature[0] == 0 &&
        out_pad.upper_size().feature[0] == 0 && out_pad.lower_size().batch[0] == 0 &&
        out_pad.upper_size().batch[0] == 0 && out_pad.lower_size().spatial[0] == 0 &&
        out_pad.lower_size().spatial[1] == 0 && out_pad.upper_size().spatial[0] == 0 &&
        out_pad.upper_size().spatial[1] == 0) {
        return true;
    }

    return false;
}

static bool can_crop_be_optimized_along_batch(const crop_node& node) {
    const auto& crop_layout = node.get_output_layout();
    auto format = crop_layout.format;
    auto input_layout = node.get_dependency(0).get_output_layout();
    const auto crop_shape = crop_layout.get_ordered_dims();
    const auto input_shape = input_layout.get_ordered_dims();
    const auto& in_padding = input_layout.data_padding;
    const auto& out_padding = crop_layout.data_padding;

    // Check format's order is 'bxxx' and only batch size is different
    if (format::is_simple_data_format(format) && format::traits(format)._order[0] == 0 &&
        std::equal(input_shape.begin()+1, input_shape.end(), crop_shape.begin()+1) &&
        !out_padding && !in_padding) {
        return true;
    }

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

// ToDo remove friendship relation from  program_node
void prepare_buffer_fusing::run(program& p) {
    /*
    We need to take care of proper ordering by types.
    1. Concats
    2. Crops
    3. Others
    Concat before crops is needed because of the crop fusing padding requirments.
    If crop is before concat there can be padding mismtach, since concat changes padding.
    */
    auto can_optimize = [](const program_node* node) {
        bool is_dynamic = node->is_dynamic();
        bool is_planar = format::is_default_format(node->get_output_layout().format);
        bool no_pad = !node->get_output_layout().data_padding && !node->get_input_layouts().empty() && !node->get_input_layout(0).data_padding;
        if (node->is_type<reshape>() && is_dynamic && is_planar && no_pad && !node->is_output() && !node->has_fused_primitives()) {
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

        // zero copy
        program_helpers::do_for_types<crop>(*node, [&p](crop_node& node) {
            // if the node is marked as network output, prevent optimizations which would affect a form of its output,
            // unless debug flag is set
            if (node.is_output())
                return;

            // do not optimize when next node is concatenation which is not output
            for (auto user : node.get_users()) {
                if (user->is_type<concatenation>() && !user->is_output())
                    return;
                if (user->is_type<loop>() || user->is_type<non_max_suppression>())
                    return;
                if (user->is_type<reshape>()) {
                    auto& reshape_node = user->as<reshape>();
                    if (can_reshape_be_optimized(reshape_node))
                        return;
                }
                if (user->is_type<experimental_detectron_roi_feature_extractor>() && user->get_dependency_index(node) == 0)
                    return;
            }

            // do not optimize crop, that must be calculated in propagate_constants
            if (node.is_constant())
                return;

            if (node.get_dependencies().size() == 1 && node.get_users().size() > 0) {
                if (p.is_body_program() && node.get_dependency(0).is_type<lstm_elt>()) {
                    return;
                }

                // optimization is available for cropping across depth(features) or batch
                // if output padding has defined padding across features already it wouldn't
                // work because it expect to have zeros in the padded area.
                if (!is_optimizable_padding_for_crop(node))
                    return;

                const auto& crop_layout = node.get_output_layout();
                const auto& crop_size = crop_layout.get_tensor();
                const auto& out_pad = crop_layout.data_padding;
                auto input_layout = node.get_input_layout(0);
                auto crop_prim = node.get_primitive();

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
                if (can_crop_be_optimized_along_feature(node)) {
                    auto crop_prim = node.get_primitive();
                    auto opt_lower_pad = crop_prim->offsets.feature[0];
                    auto opt_upper_pad = input_layout.feature() - crop_prim->offsets.feature[0] - crop_size.feature[0];
                    auto& dep = node.get_dependency(0);
                    //  feature num of pad should be accumulated if dep has been optimized out.
                    if (dep.is_type<crop>() && dep.can_be_optimized()) {
                        auto dep_pad = dep.get_output_layout().data_padding;
                        OPENVINO_ASSERT(
                            dep_pad.lower_size().batch[0] == 0 && dep_pad.upper_size().batch[0] == 0 &&
                            dep_pad.lower_size().spatial[0] == 0 && dep_pad.upper_size().spatial[0] == 0 &&
                            dep_pad.lower_size().spatial[1] == 0 && dep_pad.upper_size().spatial[1] == 0,
                            "batch, y, x of pad should be aligned to 0.");

                        opt_lower_pad += dep_pad.lower_size().feature[0];
                        opt_upper_pad += dep_pad.upper_size().feature[0];
                    }

                    // set padding
                    node.set_output_padding(
                        padding({out_pad.lower_size().batch[0],
                                opt_lower_pad,
                                out_pad.lower_size().spatial[0],
                                out_pad.lower_size().spatial[1]},
                                {out_pad.upper_size().batch[0],
                                opt_upper_pad,
                                out_pad.upper_size().spatial[0],
                                out_pad.upper_size().spatial[1]}));
                } else if (can_crop_be_optimized_along_batch(node)) {
                    auto crop_prim = node.get_primitive();
                    auto opt_lower_pad = crop_prim->offsets.batch[0];
                    auto opt_upper_pad = input_layout.batch() - crop_prim->offsets.batch[0] - crop_size.batch[0];

                    padding new_padding;
                    if (crop_layout.get_rank() == 4) {
                        new_padding = padding({opt_lower_pad,
                                    out_pad.lower_size().feature[0],
                                    out_pad.lower_size().spatial[0],
                                    out_pad.lower_size().spatial[1]},
                                    {opt_upper_pad,
                                    out_pad.upper_size().feature[0],
                                    out_pad.upper_size().spatial[0],
                                    out_pad.upper_size().spatial[1]});
                    } else if (crop_layout.get_rank() == 5) {
                        new_padding = padding({opt_lower_pad,
                                out_pad.lower_size().feature[0],
                                out_pad.lower_size().spatial[0],
                                out_pad.lower_size().spatial[1],
                                out_pad.lower_size().spatial[2]},
                                {opt_upper_pad,
                                out_pad.upper_size().feature[0],
                                out_pad.upper_size().spatial[0],
                                out_pad.upper_size().spatial[1],
                                out_pad.upper_size().spatial[2]});
                    } else {
                        return;
                    }

                    node.set_output_padding(new_padding);
                } else {
                    return;
                }

                node.can_be_optimized(true);
                propagate_padding_to_opt_out_users(node, node.get_output_layout().data_padding);
            }
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
            // If read_value here returns virable memory w/o copy, then based on Add-s and Assign execution order we may have different results
            // TODO: Allow optimizations for the case above too. Looks like it can be achieved by more careful
            // topological sort (i.e. if we ensure that all read_value users are completed before assign is run)
            node.can_be_optimized(node.get_users().size() == 1);
        });
    }
}
