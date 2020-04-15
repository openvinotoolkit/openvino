/*
// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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

#include "pass_manager.h"
#include "program_helpers.h"

#include <utility>
#include <list>
#include <vector>

using namespace cldnn;

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
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = (*node_itr++);
        if (!can_optimize(node))
            continue;
        program_helpers::do_for_types<concatenation>(*node, [&p, is_debug](concatenation_node& node) {
            // For in place concatenation input layouts and data types must match
            auto output_format = node.get_output_layout().format;
            auto output_datatype = node.get_output_layout().data_type;
            // we need to avoid mixing padded and unpadded buffer
            bool all_dependencies_padded = true;
            bool all_dependencies_unpadded = true;
            for (auto& input : node.get_dependencies()) {
                if (input->type() == reshape::type_id())
                    // reshapes should be optimized out
                    return;

                layout l = input->get_output_layout();
                if (static_cast<bool>(l.data_padding))
                    all_dependencies_unpadded = false;
                else
                    all_dependencies_padded = false;

                if (output_format != l.format || output_datatype != l.data_type)
                    return;

                if (l.format == format::b_fs_yx_fsv16 && (l.size.feature[0] % 16 != 0 || node.get_primitive()->axis != concatenation::along_f))
                    return;

                if (l.format == format::b_fs_zyx_fsv16 && (l.size.feature[0] % 16 != 0 || node.get_primitive()->axis != concatenation::along_f))
                    return;

                if ((l.format == format::b_fs_yx_fsv32 || l.format == format::b_fs_zyx_fsv32) &&
                    (l.size.feature[0] % 32 != 0 || node.get_primitive()->axis != concatenation::along_f))
                    return;

                // TODO: If we replace byxf_af32 with byxf we can probably do this optimization, but support in kernels is required
                if (l.format == format::byxf_af32 && (l.size.feature[0] % 32 != 0 || node.get_primitive()->axis != concatenation::along_f))
                    return;

                if (l.format == format::b_fs_yx_fsv4 || l.format == format::bs_fs_yx_bsv16_fsv16)
                    return;
            }

            auto concat_axis = node.get_primitive()->axis;
            auto padd = node.get_output_layout().data_padding;

            tensor lower_padd = padd.lower_size();
            tensor upper_padd = padd.upper_size();

            auto upper_padd_val =
                node.get_output_layout().get_buffer_size().raw[concat_axis] - lower_padd.raw[concat_axis];
            tensor lower_padd_offset = lower_padd;

            std::list<std::pair<const std::vector<program_node*>, tensor>> stack = {
                std::make_pair(node.get_dependencies(), tensor(0))};
            while (!stack.empty()) {
                auto nodes_list = stack.front();
                stack.pop_front();

                // if concatenation has only one input it does nothing, remove the node
                if (node.get_dependencies().size() == 1) {
                    p.extract_and_remove(node);
                    return;
                }

                auto cascade_adjustment = nodes_list.second;
                upper_padd.raw[concat_axis] = upper_padd_val;
                lower_padd = lower_padd_offset;

                auto lower_padd_in_axis = lower_padd.raw[concat_axis] + cascade_adjustment.raw[concat_axis];
                auto first_input_format = nodes_list.first[0]->get_output_layout().format;

                // check if concatenation in place can be applied for inputs set
                for (auto input : nodes_list.first) {
                    // reverted condition - if any of this node's inputs is used by more than one primitive
                    // and is not optimized concatenation then do not fuse buffers
                    // todo: we need add padding support for all optimized kernels to remove this condition
                    if (!input->is_type<pooling>() && !input->is_type<convolution>() &&
                        !input->is_type<activation>() && !input->is_type<deconvolution>() &&
                        !input->is_type<concatenation>() && !input->is_type<crop>() && !input->is_type<scale>())
                        return;

                    // if an input is marked as network output, prevent optimizations
                    // which would affect a form of its output (unless debug flag is set),
                    // we also need to restrict input types to those which support padding on all axis
                    if ((input->is_output() && !is_debug) || input->get_users().size() > 2 ||
                        !input->is_padding_supported(concat_axis, lower_padd_in_axis))
                        return;

                    if (input->get_users().size() > 1) {
                        auto user_count = input->get_users().size();
                        for (auto& user : input->get_users())
                            if (user->is_type<concatenation>())
                                user_count--;
                        if (user_count != 1)  // user_cout == 0 means that input will be used only by concatenations, so
                                              // we cannot apply concat in place for it
                            return;
                    }

                    // check if all inputs have the same format
                    if (input->get_output_layout().format != first_input_format)
                        return;

                    lower_padd_in_axis += input->get_output_layout().size.raw[concat_axis];
                }

                // check if it is worth doing concat in place, in case the following primitive is convolution
                // with different input padding than concatenation's input users' convolutions,
                // it is likely that convolution's implementation will be a reference one, due to mismatched padding
                // and performance gain by doing in place concat is nullified by slower convolution implementation
                // this should be handled by more advanced tuning mechanism on the topology level
                auto& users = node.get_users();
                if (users.size() == 1) {
                    auto& user = users.front();
                    if (node.get_output_layout().format == format::bfyx && user->type() == convolution::type_id()) {
                        auto out_input_offsets = user->as<convolution>().get_primitive()->input_offset;

                        std::vector<tensor> in_input_offsets;
                        for (auto& in_user : nodes_list.first) {
                            if (in_user->type() == convolution::type_id())
                                in_input_offsets.push_back(in_user->as<convolution>().get_primitive()->input_offset);
                        }

                        for (auto& in_input_offset : in_input_offsets) {
                            if (in_input_offset.spatial[0] != out_input_offsets.spatial[0] &&
                                in_input_offset.spatial[1] != out_input_offsets.spatial[1])
                                return;
                        }
                    } else if (user->type() == fused_conv_eltwise::type_id()) {
                        if (!user->as<fused_conv_eltwise>().get_fused_primitives().empty() &&
                            user->as<fused_conv_eltwise>().get_fused_primitives().begin()->node->is_type<depth_to_space>())
                            return;
                    }
                }

                // apply concatenation in place optimization
                for (auto input : nodes_list.first) {
                    auto input_lenght = input->get_output_layout().size.raw[concat_axis];

                    bool optimized_concat_input = false;
                    if (input->type() == concatenation::type_id() && input->can_be_optimized()) {
                        if (input->as<concatenation>().get_primitive()->axis != node.get_primitive()->axis)
                            return;
                        optimized_concat_input = true;
                    } else if (input->can_be_optimized()) {
                        return;
                    }

                    // shrink upper pad so it points at the end of the input's buffer
                    //
                    //   |--- lower padd ---|                    |---------- upper padd -----------|
                    //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
                    upper_padd.raw[concat_axis] -= input_lenght;

                    // adjust padding sizes for cascade concatenations
                    auto lower_padd_tmp = lower_padd;
                    lower_padd_tmp.raw[concat_axis] += cascade_adjustment.raw[concat_axis];
                    auto upper_padd_tmp = upper_padd;
                    upper_padd_tmp.raw[concat_axis] -= cascade_adjustment.raw[concat_axis];

                    // set new padding for input
                    input->set_output_padding(padding(lower_padd_tmp.sizes(), upper_padd_tmp.sizes()));

                    // move lower padd further
                    //
                    //   |-------------- lower padd -------------|---------- upper padd -----------|
                    //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|

                    lower_padd.raw[concat_axis] += input_lenght;

                    if (optimized_concat_input && !input->get_dependencies().empty())
                        stack.push_back(std::make_pair(input->get_dependencies(),
                                                       input->get_output_layout().data_padding.lower_size()));
                }
            }

            node.can_be_optimized(true);
            for (auto dep : node.get_users()) {
                dep->can_share_buffer(false);
            }
            if (!all_dependencies_padded && !all_dependencies_unpadded)
                node.can_share_buffer(false);
        });
    }

    // [2] Then try to optimize all crops
    node_itr = p.get_processing_order().begin();
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
            }

            if (node.get_dependencies().size() == 1 && node.get_users().size() > 0) {
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
