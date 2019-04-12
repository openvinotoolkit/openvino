/*
// Copyright (c) 2018 Intel Corporation
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

#include "api/CPP/eltwise.hpp"
#include "api/CPP/pooling.hpp"
#include "api/CPP/upsampling.hpp"
#include "primitive_inst.h"
#include "activation_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "reshape_inst.h"
#include "scale_inst.h"

#include "pass_manager.h"
#include "program_helpers.h"


using namespace cldnn;

//ToDo remove friendship relation from  program_node 

void prepare_buffer_fusing::run(program_impl& p)
{
    bool is_debug = p.get_options().get<build_option_type::debug>()->enabled();
    /*
    We need to take care of proper ordering by types.
    1. Concats
    2. Crops
    3. Others
    Concat before crops is needed because of the crop fusing padding requirments. 
    If crop is before concat there can be padding mismtach, since concat changes padding.
    */
    auto can_optimize = [](const program_node* node)
    {
        if (node->is_output() ||
            (node->get_fused_activation_func() != cldnn_activation_func_t::activation_none))
        {
            return false;
        }
        return true;
    };

    //[1] First try to optimize all concats
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end())
    {
        auto& node = (*node_itr++);
        if (!can_optimize(node))
            continue;
        program_helpers::do_for_types<concatenation>(*node, [&p, is_debug](concatenation_node& node)
        {
            // we need to avoid mixing padded and unpadded buffer 
            bool all_dependencies_padded = true;
            bool all_dependencies_unpadded = true;
            for (auto& input : node.get_dependencies()) {
                layout l = input->get_output_layout();
                if (static_cast<bool>(l.data_padding))
                    all_dependencies_unpadded = false;
                else
                    all_dependencies_padded = false;
            }
            auto concat_axis = node.get_primitive()->axis;
            auto padd = node.get_output_layout().data_padding;

            tensor lower_padd = padd.lower_size();
            tensor upper_padd = padd.upper_size();

            auto upper_padd_val = node.get_output_layout().get_buffer_size().raw[concat_axis] - lower_padd.raw[concat_axis];
            tensor lower_padd_offset = lower_padd;

            std::list<std::pair<const std::vector<program_node*>, tensor>> stack = { std::make_pair(node.get_dependencies(), tensor{ 0, 0, 0, 0 }) };
            while (!stack.empty())
            {
                auto nodes_list = stack.front();
                stack.pop_front();

                auto cascade_adjustment = nodes_list.second;
                upper_padd.raw[concat_axis] = upper_padd_val;
                lower_padd = lower_padd_offset;

                //check if concatenation in place can be applied for inputs set
                for (auto input : nodes_list.first)
                {
                    //if any of this node's inputs is used by more than one primitive and is not optimized concatenation then do not fuse buffers,
                    //also, if an input is marked as network output, prevent optimizations which would affect a form of its output (unless debug flag is set)
                    // todo: in future, if this case is problem, it can be optimized further to enable buffer fusing
                    //       per single input rather than all/none
                    // + restrict input types to those which support padding on x,y,b and f
                    if (!input->support_padding() ||
                        (input->is_output() && !is_debug) ||
                        input->get_users().size() > 2)
                        return;

                    if (input->get_users().size() > 1)
                    {
                        auto user_count = input->get_users().size();
                        for (auto& user : input->get_users())
                            if (user->is_type<concatenation>())
                                user_count--;
                        if (user_count != 1) // user_cout == 0 means that input will be used only by concatenations, so we cannot apply concat in place for it
                            return;
                    }
                }

                //apply concatenation in place optimization
                for (auto input : nodes_list.first)
                {
                    auto input_lenght = input->get_output_layout().size.raw[concat_axis];

                    bool optimized_concat_input = false;
                    if (input->type() == concatenation::type_id() && input->can_be_optimized())
                    {
                        if (input->as<concatenation>().get_primitive()->axis != node.get_primitive()->axis)
                            return;
                        optimized_concat_input = true;
                    }

                    // shrink upper pad so it points at the end of the input's buffer
                    //
                    //   |--- lower padd ---|                    |---------- upper padd -----------|
                    //   |-- output padd ---| ----- input1 ------|----- input2 -----|-- out padd --|
                    upper_padd.raw[concat_axis] -= input_lenght;

                    //adjust padding sizes for cascade concatenations
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
                        stack.push_back(std::make_pair(input->get_dependencies(), input->get_output_layout().data_padding.lower_size()));
                }
            }

            node.can_be_optimized(true);
            for (auto dep : node.get_users())
            {
                dep->can_share_buffer(false);
            }
            if (!all_dependencies_padded && !all_dependencies_unpadded)
                node.can_share_buffer(false);
        });
    }

    //[2] Then try to optimize all crops
    node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end())
    {
        auto& node = (*node_itr++);
        if (!can_optimize(node))
            continue;
        // zero copy
        program_helpers::do_for_types<crop>(*node, [&p, is_debug](crop_node& node)
        {
            //if the node is marked as network output, prevent optimizations which would affect a form of its output, unless debug flag is set
            if (node.is_output() && !is_debug)
                return;

            //do not optimize when next node is concatenation which is not output
            if (node.get_users().size() == 1 && node.get_users().front()->is_type<concatenation>() && !node.get_users().front()->is_output())
                return;

            if (node.get_dependencies().size() == 1 &&
                node.get_users().size() > 0)
            {
                // optimization is available for cropping across depth(features) only
                // if output padding has defined padding across features already it wouldn't
                // work because it expect to have zeros in the padded area.
                const auto& crop_layout = node.get_output_layout();
                auto format = crop_layout.format;
                auto crop_prim = node.get_primitive();
                auto input_layout = node.get_dependency(0).get_output_layout();
                const auto& crop_size = crop_layout.size;
                const auto& out_padd = crop_layout.data_padding;
                if (format == format::bfyx &&
                    crop_size.batch[0] == input_layout.size.batch[0] &&
                    crop_size.spatial[0] == input_layout.size.spatial[0] &&
                    crop_size.spatial[1] == input_layout.size.spatial[1] &&
                    out_padd.lower_size().feature[0] == 0 &&
                    out_padd.upper_size().feature[0] == 0 &&
                    out_padd.lower_size().batch[0] == 0 &&
                    out_padd.upper_size().batch[0] == 0 &&
                    out_padd.lower_size().spatial[0] == 0 &&
                    out_padd.lower_size().spatial[1] == 0 &&
                    out_padd.upper_size().spatial[0] == 0 &&
                    out_padd.upper_size().spatial[1] == 0)
                {
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

                    node.set_output_padding(padding(
                        { out_padd.lower_size().batch[0], crop_prim->offsets.feature[0], out_padd.lower_size().spatial[0], out_padd.lower_size().spatial[1] },
                        { out_padd.upper_size().batch[0], input_layout.size.feature[0] - crop_prim->offsets.feature[0] - crop_size.feature[0],
                            out_padd.upper_size().spatial[0], out_padd.upper_size().spatial[1] }));
                    node.can_be_optimized(true);
                }
            }
        });
    }

    //[3] Optimize all other primitives
    node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end())
    {
        auto& node = (*node_itr++);
        if (!can_optimize(node))
            continue;
        program_helpers::do_for_types<reshape>(*node, [&p](reshape_node& node)
        {
            node.get_output_layout();
            if (node.is_in_place()
                && node.get_fused_activation_func() == activation_none)
                node.can_be_optimized(true);
        });
        program_helpers::do_for_types<reorder>(*node, [&p](reorder_node& node)
        {
            auto& input = node.input();
            auto output_layout = node.get_output_layout();
            //This is WA for topologies that due to additional reorders added perform worse with conv1x1 optimization
            auto remove_bf8_xy_opt = ((input.is_type<pooling>() || input.is_type<concatenation>()) &&
                output_layout.format == format::bf8_xy16 && input.get_users().size() == 1);
            //Remove reorder from convolution 1x1 to bfyx in some conditions
            auto remove_byxf_opt = (input.is_type<convolution>() &&
                input.get_users().size() == 1 &&
                input.get_output_layout().format == format::byxf);
            //check if all inputs user have the same format
            auto all_users_same_format = true;
            auto input_user_layout_format = input.get_users().front()->get_output_layout().format;
            for (auto const& user : input.get_users())
            {
                if (user->get_output_layout().format != input_user_layout_format)
                {
                    all_users_same_format = false;
                    break;
                }
            }
            auto same_data_type = input.get_output_layout().data_type == output_layout.data_type;
            //Optimization only available in case of layers that support different input and output formats.
            //todo: new api needs to be created to read such caps
            if (!(input.is_type<pooling>() && (output_layout.format == format::bfyx || output_layout.format == format::yxfb || output_layout.format == format::byxf) && all_users_same_format && same_data_type) &&
                !remove_bf8_xy_opt &&
                !(input.is_type<convolution>() && input.get_output_layout().format == format::bf8_xy16) &&
                !(input.is_type<eltwise>() && (output_layout.format == format::bfyx || output_layout.format == format::yxfb || output_layout.format == format::byxf) && all_users_same_format && same_data_type) &&
                !(remove_byxf_opt && (node.get_users().front()->is_type<eltwise>() || node.get_users().front()->is_type<pooling>())))
                return;

            if (remove_bf8_xy_opt)
            {
                auto users_user_layout = node.get_users().front()->get_users().front()->get_output_layout();
                // if users_user_layout is still bf8_yx16 (stacked convolutions) then leave the reorder
                if (users_user_layout.format == format::bf8_xy16)
                    return;
                auto input_layout = input.get_output_layout();
                auto target_layout = layout(input_layout.data_type, users_user_layout.format, input_layout.size, input_layout.data_padding);
                input.set_output_layout(target_layout, false);
            }
            else if (remove_byxf_opt)
            {
                auto user = node.get_users().front();
                auto users_users = node.get_users().front()->get_users();

                for (auto const& users_user : users_users)
                {
                    if (users_user->get_output_layout().format != format::byxf && !users_user->is_type<eltwise>())
                    {
                        remove_byxf_opt = false;
                        break;
                    }
                }

                if (remove_byxf_opt)
                {
                    auto input_layout = input.get_output_layout();
                    user->set_output_layout(input_layout, false);
                }
            }
            else
                input.set_output_layout(output_layout, false);

            node.can_be_optimized(true);
            p.extract_and_remove(node); //try to remove redundant reorders
        });
    }
}
