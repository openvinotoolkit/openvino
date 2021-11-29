// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/eltwise.hpp"
#include "api/pooling.hpp"
#include "fused_conv_eltwise_inst.h"
#include "primitive_inst.h"
#include "activation_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "reshape_inst.h"
#include "scale_inst.h"
#include "depth_to_space_inst.h"
#include "resample_inst.h"
#include "loop_inst.h"
#include "non_max_suppression_inst.h"

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
    if (node.is_output() && !get_program().is_debug_build())
        return false;
    return node.get_dependencies().size() == 1 &&
        !node.has_fused_primitives() &&
        node.get_fused_activations_funcs().empty();
}

bool concat_noop_optimization::optimize(concatenation_node& node) {
    auto& dep = node.get_dependency(0);
    dep.merge_output_padding(node.get_output_layout().data_padding);
    prog.extract_and_remove(node);
    // Node has been removed, so no further optimizations.
    return true;
}

bool concat_in_place_optimization::match(concatenation_node& node) {
    if (node.is_output() && !get_program().is_debug_build())
        return false;
    if (node.has_fused_primitives() || !node.get_fused_activations_funcs().empty())
        return false;

    // For in place concatenation input layouts and data types must match.
    auto output_format = node.get_output_layout().format;
    auto output_datatype = node.get_output_layout().data_type;
    auto concat_axis = node.get_primitive()->axis;

    for (auto& input : node.get_dependencies()) {
        if (input->is_type<reshape>())
            // reshapes should be optimized out.
            return false;

        layout l = input->get_output_layout();

        if (output_format != l.format || output_datatype != l.data_type)
            return false;

        // TODO: Below condition should be moved to program_node::supports_padding.
        // This hovewer will need updating the algorithm as it may make cascade adjustment impossible in some cases.
        // It hovewer would make normal optimizations possible in others, so this is a trade-off to be investigated.
        if (l.format == format::b_fs_yx_fsv16 && (l.size.feature[0] % 16 != 0 || node.get_primitive()->axis != concatenation::along_f))
            return false;

        if (l.format == format::b_fs_zyx_fsv16 && (l.size.feature[0] % 16 != 0 || node.get_primitive()->axis != concatenation::along_f))
            return false;

        if ((l.format == format::b_fs_yx_fsv32 || l.format == format::b_fs_zyx_fsv32) &&
            (l.size.feature[0] % 32 != 0 || node.get_primitive()->axis != concatenation::along_f))
            return false;

        if (l.format == format::bs_fs_yx_bsv16_fsv16)
            return false;

        if (l.format == format::b_fs_yx_fsv4 && (l.size.feature[0] != 8 || node.get_primitive()->axis != concatenation::along_f))
            return false;
    }

    auto lower_padd_in_axis = node.get_output_layout().data_padding.lower_size().raw[concat_axis];
    lower_padd_in_axis = std::max(lower_padd_in_axis,
                                  node.get_dependency(0).get_output_layout().data_padding.lower_size().raw[concat_axis]);

    // check if concatenation in place can be applied for inputs set
    size_t idx = 0;
    for (auto input : node.get_dependencies()) {
        // reverted condition - if any of this node's inputs is used by more than one primitive
        // and is not optimized concatenation then do not fuse buffers
        // todo: we need add padding support for all optimized kernels to remove this condition
        if (!input->is_type<pooling>() && !input->is_type<convolution>() &&
            !input->is_type<activation>() && !input->is_type<deconvolution>() &&
            !input->is_type<concatenation>() && !input->is_type<crop>() && !input->is_type<scale>() && !input->is_type<eltwise>() &&
            !input->is_type<resample>())
            return false;

        // if an input is marked as network output, prevent optimizations
        // which would affect a form of its output (unless debug flag is set),
        // we also need to restrict input types to those which support padding on all axis
        if ((input->is_output() && !get_program().is_debug_build()) ||
            !input->is_padding_supported(concat_axis, lower_padd_in_axis))
            return false;

        // TODO: Investigate if this condition is needed
        if (input->get_users().size() > 2)
            return false;

        // Check that input isn't optimized out concatenation along different axis.
        if (input->is_type<concatenation>() && input->can_be_optimized() &&
            input->as<concatenation>().get_primitive()->axis != concat_axis)
            return false;

        // Check that input isn't optimized out non-concatenation.
        if (!input->is_type<concatenation>() && input->can_be_optimized())
            return false;

        size_t concat_users = 0;
        for (auto& user : input->get_users())
            if (user->is_type<concatenation>())
                concat_users += 1;

        // If input is used by more than one concatenation then they may require different paddings.
        if (concat_users != 1)
            return false;

        auto input_padd = input->get_output_layout().data_padding;

        // Check that there isn't already some padding between inputs in concat axis.
        // If node has already been optimized we skip this check - this is just cascade adjustment.
        if (!node.can_be_optimized()) {
            if (idx != node.get_dependencies().size() && input_padd.upper_size().raw[concat_axis] != 0)
                return false;
            if (idx != 0 && input_padd.lower_size().raw[concat_axis] != 0)
                return false;
        }

        lower_padd_in_axis += input->get_output_layout().size.raw[concat_axis];
        idx += 1;
    }

    return true;
}

void concat_in_place_optimization::optimize_cascade(concatenation_node& node, std::list<concatenation_node*>& need_reoptimization) {
    auto concat_axis = node.get_primitive()->axis;

    // Select output padding by propagating all required input paddings.
    auto padd = node.get_output_layout().data_padding;
    for (auto input : node.get_dependencies()) {
        padd = padding::max(padd, input->get_output_layout().data_padding);
    }

    auto lower_padd = padd.lower_size();
    auto upper_padd = padd.upper_size();

    // For cascade adjustment override padding in concat axis to output padding.
    // In other case match(...) already checked that only first/last input have lower/upper padding.
    if (node.can_be_optimized()) {
        lower_padd.raw[concat_axis] = node.get_output_layout().data_padding.lower_size().raw[concat_axis];
        upper_padd.raw[concat_axis] = node.get_output_layout().data_padding.upper_size().raw[concat_axis];
    }
    node.set_output_padding(padding(lower_padd.sizes(), upper_padd.sizes()));

    upper_padd.raw[concat_axis] += node.get_output_layout().size.raw[concat_axis];

    // apply concatenation in place optimization
    for (auto input : node.get_dependencies()) {
        auto input_length = input->get_output_layout().size.raw[concat_axis];

        if (input->is_type<concatenation>() && input->can_be_optimized())
            need_reoptimization.push_back(&input->as<concatenation>());

        // shrink upper pad so it points at the end of the input's buffer
        //
        //   |--- lower padd ---|                    |---------- upper padd -----------|
        //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
        upper_padd.raw[concat_axis] -= input_length;

        // set new padding for input
        input->set_output_padding(padding(lower_padd.sizes(), upper_padd.sizes()));

        // move lower padd further
        //
        //   |-------------- lower padd -------------|---------- upper padd -----------|
        //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
        lower_padd.raw[concat_axis] += input_length;
    }

    node.can_be_optimized(true);
    for (auto dep : node.get_users()) {
        dep->can_share_buffer(false);
    }
}

}  // namespace

// ToDo remove friendship relation from  program_node
void prepare_buffer_fusing::run(program_impl& p) {
    bool is_debug = p.get_options().get<build_option_type::debug>()->enabled();
    /*
    We need to take care of proper ordering by types.
    1. Concats
    2. Crops
    3. Others
    Concat before crops is needed because of the crop fusing padding requirments.
    If crop is before concat there can be padding mismtach, since concat changes padding.
    */
    auto can_optimize = [](const program_node* node) {
        if (node->is_output() || (!node->get_fused_activations_funcs().empty())) {
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
        if (!can_optimize(node))
            continue;
        // zero copy
        program_helpers::do_for_types<crop>(*node, [&p, is_debug](crop_node& node) {
            // if the node is marked as network output, prevent optimizations which would affect a form of its output,
            // unless debug flag is set
            if (node.is_output() && !is_debug)
                return;

            // do not optimize when next node is concatenation which is not output
            for (auto user : node.get_users()) {
                if (user->is_type<concatenation>() && !user->is_output())
                    return;
                if (user->is_type<loop>() || user->is_type<non_max_suppression>())
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
                const auto& crop_size = crop_layout.size;
                const auto& out_padd = crop_layout.data_padding;
                const auto opt_lower_pad = crop_prim->offsets.feature[0];
                const auto opt_upper_pad = input_layout.size.feature[0] - crop_prim->offsets.feature[0] - crop_size.feature[0];

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
                }

                if (format == format::bfyx && crop_size.batch[0] == input_layout.size.batch[0] &&
                    crop_size.spatial[0] == input_layout.size.spatial[0] &&
                    crop_size.spatial[1] == input_layout.size.spatial[1] && out_padd.lower_size().feature[0] == 0 &&
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
        if (!can_optimize(node))
            continue;
        program_helpers::do_for_types<reshape>(*node, [&p](reshape_node& node) {
            node.get_output_layout();
            if (node.is_in_place() && node.get_fused_activations_funcs().empty())
                node.can_be_optimized(true);
            else
                node.can_be_optimized(false);
        });
    }
}
