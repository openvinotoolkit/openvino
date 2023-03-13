// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "primitive_inst.h"
#include "activation_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "reshape_inst.h"
#include "depth_to_space_inst.h"
#include "resample_inst.h"
#include "loop_inst.h"
#include "non_max_suppression_inst.h"
#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "border_inst.h"

#include "pass_manager.h"
#include "program_helpers.h"

#include <utility>
#include <list>
#include <vector>

using namespace cldnn;

namespace {

struct concat_noop_optimization : pattern_match_optimization_typed<concat_noop_optimization, concatenation> {
    // Removes concatenation nodes with single input.
    using base = pattern_match_optimization_typed<concat_noop_optimization, concatenation>;
    using base::base;

    bool match(concatenation_node& node);
    bool optimize(concatenation_node& node);
};

struct concat_in_place_optimization : pattern_match_optimization_typed<concat_in_place_optimization, concatenation> {
    // Performs in-place concat optimization.
    // Padding of predecessors is updated to use single buffer by all, which is output from concatenation.
    // Then concatenation can be optimized out, as memory will be correctly filled by previous nodes.
    // If one of the dependencies is also optimized-out concatenation, then cascade adjusment is performed to update it.
    // This optimization is expected to be executed in some topological order, as cascade adjustment is performed backwards.
    using base = pattern_match_optimization_typed<concat_in_place_optimization, concatenation>;
    using base::base;

    // Runs concat in-place optimization and adds already optimized concatenations that need re-optimization to `needs_reoptimization`.
    void optimize_cascade(concatenation_node& node, std::list<concatenation_node*>& need_reoptimization);
    bool match(concatenation_node& node);
    bool optimize(concatenation_node& node) {
        std::list<concatenation_node*> need_reopt;
        optimize_cascade(node, need_reopt);
        while (!need_reopt.empty()) {
            auto& prop = *need_reopt.front();
            need_reopt.pop_front();
            if (match(prop))
                optimize_cascade(prop, need_reopt);
            else
                // TODO: Revert extra padding when cascade adjustment failed.
                prop.can_be_optimized(false);
        }
        return false;  // node not invalidated
    }
};

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
    if (node.is_output())
        return false;
    if (node.has_fused_primitives())
        return false;
    if (node.is_dynamic())
        return false;

    bool is_onednn_impl = false;

    for (auto& input : node.get_dependencies()) {
        if (input.first->get_preferred_impl_type() == impl_types::onednn) {
            for (auto& fused_op : input.first->get_fused_primitives()) {
                auto add_type = onednn_add_fusing_helpers::get_add_fusing_type(*input.first, fused_op);
                if (add_type == add_fusing_type::sum)
                    return false;
                else
                    continue;
            }

            // Optimized-out input node is no longer onednn impl.
            if (!input.first->can_be_optimized())
                is_onednn_impl = true;
        }
    }

    // Implicit concat for onednn only when use_usm and batch 1.
    if (is_onednn_impl) {
        bool use_usm = node.get_program().get_engine().use_unified_shared_memory();
        layout out_l = node.get_output_layout();

        if (!use_usm)
            return false;
        if (out_l.batch() > 1)
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
        if (std::find_if(white_list.begin(), white_list.end(), [&out_l](format fmt){ return (fmt == out_l.format); }) == std::end(white_list))
            return false;
    }

    // For in place concatenation input layouts and data types must match.
    // Also, it checks whether data along f-axis is aligned properly for implicit concat.
    // Otherwise, use explicit concat instead.
    auto output_format = node.get_output_layout().format;
    auto output_datatype = node.get_output_layout().data_type;
    auto concat_axis = node.get_primitive()->axis;
    auto def_fmt = format::get_default_format(node.get_output_layout().get_rank());

    size_t idx = 0;
    for (auto& input : node.get_dependencies()) {
        if (input.first->is_type<reshape>())
            // reshapes should be optimized out.
            return false;

        layout l = input.first->get_output_layout();

        if (output_format != l.format || output_datatype != l.data_type)
            return false;

        if (l.format.block_sizes().size() > 1)
            return false;

        // TODO: Below condition should be moved to program_node::supports_padding.
        // This however will need updating the algorithm as it may make cascade adjustment impossible in some cases.
        // It however would make normal optimizations possible in others, so this is a trade-off to be investigated.
        if (idx != node.get_dependencies().size() - 1) {
            if ((l.format == format::b_fs_yx_fsv16 || l.format == format::b_fs_zyx_fsv16) &&
                (l.feature() % 16 != 0 || node.get_primitive()->axis != 1))
                return false;

            if ((l.format == format::b_fs_yx_fsv32 || l.format == format::b_fs_zyx_fsv32) &&
                (l.feature() % 32 != 0 || node.get_primitive()->axis != 1))
                return false;

            if (l.format == format::b_fs_yx_fsv4 && (l.feature() != 4 || node.get_primitive()->axis != 1))
                return false;
        }
        idx++;
    }

    auto lower_padd_in_axis = node.get_output_layout().data_padding.lower_size().sizes(def_fmt)[concat_axis];
    lower_padd_in_axis = std::max(lower_padd_in_axis,
                                  node.get_dependency(0).get_output_layout().data_padding.lower_size().sizes(def_fmt)[concat_axis]);

    // check if concatenation in place can be applied for inputs set
    idx = 0;
    for (auto input : node.get_dependencies()) {
        // reverted condition - if any of this node's inputs is used by more than one primitive
        // and is not optimized concatenation then do not fuse buffers
        // todo: we need add padding support for all optimized kernels to remove this condition
        if (!input.first->is_type<pooling>() && !input.first->is_type<convolution>() && !input.first->is_type<quantize>() &&
            !input.first->is_type<activation>() && !input.first->is_type<deconvolution>() &&
            !input.first->is_type<concatenation>() && !input.first->is_type<crop>() && !input.first->is_type<eltwise>() &&
            !input.first->is_type<resample>())
            return false;

        // if an input is marked as network output, prevent optimizations
        // which would affect a form of its output (unless debug flag is set),
        // we also need to restrict input types to those which support padding on all axis
        if (input.first->is_output() || !input.first->is_padding_supported(concat_axis, lower_padd_in_axis))
            return false;

        // TODO: Investigate if this condition is needed
        if (input.first->get_users().size() > 2)
            return false;

        // If sibling is using onednn impl and batch > 1, the onednn impl cannot process the implicit concat'ed buffer.
        // Onednn impls can process implicit concat'ed buffer only through buffer pointer manipulation.
        if (node.get_output_layout().batch() > 1) {
            for (auto& sib : input.first->get_users()) {
                if (sib->get_preferred_impl_type() == impl_types::onednn) {
                    return false;
                }
            }
        }

        // Check that input isn't optimized out concatenation along different axis.
        if (input.first->is_type<concatenation>() && input.first->can_be_optimized() &&
            input.first->as<concatenation>().get_primitive()->axis != concat_axis)
            return false;

        // Check that input isn't optimized out non-concatenation.
        if (!input.first->is_type<concatenation>() && input.first->can_be_optimized())
            return false;

        size_t concat_users = 0;
        for (auto& user : input.first->get_users())
            if (user->is_type<concatenation>())
                concat_users += 1;

        // If input is used by more than one concatenation then they may require different paddings.
        if (concat_users != 1)
            return false;

        auto input_padd = input.first->get_output_layout().data_padding;

        // Check that there isn't already some padding between inputs in concat axis.
        // If node has already been optimized we skip this check - this is just cascade adjustment.
        if (!node.can_be_optimized()) {
            if (idx != node.get_dependencies().size() && input_padd.upper_size().sizes(def_fmt)[concat_axis] != 0)
                return false;
            if (idx != 0 && input_padd.lower_size().sizes(def_fmt)[concat_axis] != 0)
                return false;
        }

        lower_padd_in_axis += input.first->get_output_layout().get_tensor().sizes(def_fmt)[concat_axis];
        idx += 1;
    }

    return true;
}

void concat_in_place_optimization::optimize_cascade(concatenation_node& node, std::list<concatenation_node*>& need_reoptimization) {
    auto out_layout = node.get_output_layout();
    auto out_rank = out_layout.get_rank();
    auto concat_axis = node.get_primitive()->axis;
    // We need to transform axis from bf[w][z]yx order to bfxy[z][w] due to tensor.sizes() usages here
    // should be removed once pad representation is changed
    auto concat_axis_legacy = concat_axis;
    if (concat_axis_legacy >= 2) {
        auto spatial_axis = concat_axis_legacy - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max<size_t>(out_rank, 4) - 2;
        concat_axis_legacy = spatial_size - spatial_axis - 1 + 2;
    }

    // Select output padding by propagating all required input paddings.
    auto padd = out_layout.data_padding;
    for (auto input : node.get_dependencies()) {
        auto inputPadding = input.first->get_output_layout().data_padding;
        padd = padding::max(padd, inputPadding);
    }

    auto lower_padd = padd.lower_size().sizes();
    auto upper_padd = padd.upper_size().sizes();

    // For cascade adjustment override padding in concat axis to output padding.
    // In other case match(...) already checked that only first/last input have lower/upper padding.
    if (node.can_be_optimized()) {
        lower_padd[concat_axis_legacy] = out_layout.data_padding.lower_size().sizes()[concat_axis_legacy];
        upper_padd[concat_axis_legacy] = out_layout.data_padding.upper_size().sizes()[concat_axis_legacy];
    }
    node.set_output_padding(padding(lower_padd, upper_padd));

    upper_padd[concat_axis_legacy] += out_layout.get_dims()[concat_axis];

    // apply concatenation in place optimization
    for (auto input : node.get_dependencies()) {
        auto input_length = input.first->get_output_layout().get_dims()[concat_axis];

        if (input.first->is_type<concatenation>() && input.first->can_be_optimized())
            need_reoptimization.push_back(&input.first->as<concatenation>());

        // shrink upper pad so it points at the end of the input's buffer
        //
        //   |--- lower padd ---|                    |---------- upper padd -----------|
        //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
        upper_padd[concat_axis_legacy] -= input_length;

        // set new padding for input
        input.first->set_output_padding(padding(lower_padd, upper_padd));

        // move lower padd further
        //
        //   |-------------- lower padd -------------|---------- upper padd -----------|
        //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
        lower_padd[concat_axis_legacy] += input_length;
    }

    node.can_be_optimized(true);
    for (auto dep : node.get_users()) {
        dep->can_share_buffer(false);
    }
}

}  // namespace

static bool can_reshape_be_optimized(const reshape_node& node) {
    return node.is_in_place() && !node.has_fused_primitives();
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
        bool is_dynamic = node->get_output_layout().is_dynamic();
        bool is_planar = node->get_output_layout().format == format::bfyx ||
                         node->get_output_layout().format == format::bfzyx ||
                         node->get_output_layout().format == format::bfwzyx;
        bool no_pad = !node->get_output_layout().data_padding && !node->get_input_layouts().empty() && !node->get_input_layouts()[0].data_padding;
        // The condition below check only output layout as cases like
        // (dyn_shape) -> reshape -> (static_shape) -> some_static_primitive
        // may have invalid set_arguments call as output memory of reshape won't be available until reshape primitive is executed
        if (node->is_type<reshape>() && is_dynamic && is_planar && no_pad && !node->is_output() && !node->has_fused_primitives()) {
            return true;
        }

        if (node->is_dynamic() || node->is_output() || node->has_fused_primitives()) {
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

            if (node.get_dependencies().size() == 1 && node.get_users().size() > 0) {
                if (p.is_loop_body() && node.get_dependency(0).is_type<lstm_elt>()) {
                    return;
                }
                // optimization is available for cropping across depth(features) only
                // if output padding has defined padding across features already it wouldn't
                // work because it expect to have zeros in the padded area.
                const auto& crop_layout = node.get_output_layout();
                auto format = crop_layout.format;
                auto crop_prim = node.get_primitive();
                auto input_layout = node.get_dependency(0).get_output_layout();
                const auto& crop_size = crop_layout.get_tensor();
                const auto& out_padd = crop_layout.data_padding;
                auto opt_lower_pad = crop_prim->offsets.feature[0];
                auto opt_upper_pad = input_layout.feature() - crop_prim->offsets.feature[0] - crop_size.feature[0];

                // do not optimize crop if paddings are not properly aligned
                for (auto& usr : node.get_users()) {
                    auto usr_layout = usr->get_output_layout();
                    if (usr_layout.format == format::b_fs_yx_fsv16 &&
                        (opt_lower_pad % 16 != 0 || opt_upper_pad % 16 != 0))
                        return;
                    if (input_layout.data_padding.lower_size().batch[0] != 0 || input_layout.data_padding.upper_size().batch[0] != 0 ||
                        input_layout.data_padding.lower_size().spatial[0] != 0 || input_layout.data_padding.upper_size().spatial[0] != 0 ||
                        input_layout.data_padding.lower_size().spatial[1] != 0 || input_layout.data_padding.upper_size().spatial[1] != 0)
                        return;
                    // oneDNN doesn't support paddings
                    if (usr->get_preferred_impl_type() == impl_types::onednn)
                        return;
                }

                if (format == format::bfyx && crop_size.batch[0] == input_layout.batch() &&
                    crop_size.spatial[0] == input_layout.spatial(0) &&
                    crop_size.spatial[1] == input_layout.spatial(1) && out_padd.lower_size().feature[0] == 0 &&
                    out_padd.upper_size().feature[0] == 0 && out_padd.lower_size().batch[0] == 0 &&
                    out_padd.upper_size().batch[0] == 0 && out_padd.lower_size().spatial[0] == 0 &&
                    out_padd.lower_size().spatial[1] == 0 && out_padd.upper_size().spatial[0] == 0 &&
                    out_padd.upper_size().spatial[1] == 0) {
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

                    //  feature num of pad should be accumulated if dep has been optimized out.
                    auto& dep = node.get_dependency(0);
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

                    node.set_output_padding(
                        padding({out_padd.lower_size().batch[0],
                                 opt_lower_pad,
                                 out_padd.lower_size().spatial[0],
                                 out_padd.lower_size().spatial[1]},
                                {out_padd.upper_size().batch[0],
                                 opt_upper_pad,
                                 out_padd.upper_size().spatial[0],
                                 out_padd.upper_size().spatial[1]}));
                    node.can_be_optimized(true);
                }
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
            node.can_be_optimized(can_reshape_be_optimized(node));
        });
    }
}
