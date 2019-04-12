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

#include "api/CPP/pooling.hpp"
#include "api/CPP/proposal.hpp"
#include "api/CPP/roi_pooling.hpp"

#include "program_helpers.h"
#include "pass_manager.h"

#include "activation_inst.h"
#include "batch_norm_inst.h"
#include "batch_norm_grad_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "fused_conv_bn_scale_inst.h"
#include "fused_conv_eltwise_inst.h"
#include "lrn_inst.h"
#include "mutable_data_inst.h"
#include "mvn_inst.h"
#include "normalize_inst.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "softmax_inst.h"
#include "scale_inst.h"
#include "scale_grad_weights_inst.h"
#include "upsampling_inst.h"


void prepare_primitive_fusing::fuse_skip_layers(program_impl& p, program_node* node)
{
    program_helpers::do_for_types<eltwise>(*node, [&p](eltwise_node& node)
    {
        if (node.get_primitive()->mode != eltwise_mode::sum || node.inputs_count() != 2)
            return;

        // both inputs should be deconvolutions
        if (!(node.input(0).is_type<deconvolution>() && node.input(1).is_type<deconvolution>()))
        {
            return;
        }

        auto& to_fuse_with = node.input(0);
        int to_fuse_index = 1;

        //remove dependencies and users of elwtise that is going to be extracted
        p.add_connection(node.input(to_fuse_index), to_fuse_with);
        p.remove_connection(node.input(to_fuse_index), node);

        p.get_processing_order().erase(&to_fuse_with);
        p.get_processing_order().insert(&node, &to_fuse_with);

        if (node.get_fused_activation_func() != activation_none)
            to_fuse_with.set_fused_activation(node.get_fused_activation_func(), node.get_fused_activation_params());
        to_fuse_with.set_output_padding(node.get_output_layout().data_padding);

        p.extract_and_remove(node);
    });
}

template<typename T>
static bool node_is_type(program_node* n)
{
    return n->is_type<T>();
}

void prepare_primitive_fusing::fuse_conv_bn_scale(program_impl& p, program_node* node)
{
    program_helpers::do_for_types<convolution>(*node, [&p](convolution_node& node)
    {
        if (node.get_users().size() > 2)
            return;

        auto found_bn = std::find_if(node.get_users().begin(), node.get_users().end(), node_is_type<batch_norm>);
        auto bn_node = found_bn != node.get_users().end() ? *found_bn : nullptr;
        if (bn_node != nullptr)
        {
            if (bn_node->get_users().size() > 2)
                return;

            auto found_scale = std::find_if(bn_node->get_users().begin(), bn_node->get_users().end(), node_is_type<scale>);
            auto sc_node = found_bn != node.get_users().end() ? *found_scale : nullptr;
            if (sc_node != nullptr)
            {
                int bn_index = int(std::distance(node.get_users().begin(), found_bn));
                int sc_index = int(std::distance(bn_node->get_users().begin(), found_scale));
                auto scale_prim = std::static_pointer_cast<const scale>(sc_node->get_primitive());
                auto bn_prim = std::static_pointer_cast<const batch_norm>(bn_node->get_primitive());
                auto prim = node.get_primitive();
                bool training = false;

                if (node.get_users().size() == 2)
                {
                    training = true;
                    float zero = 0.0f;
                    layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));

                    auto bn_backw = node.get_users().begin();
                    std::advance(bn_backw, bn_index == 0 ? 1 : 0);
                    if (!(*bn_backw)->is_type<batch_norm_grad>())
                        return;
                    auto sc_backw = bn_node->get_users().begin();
                    std::advance(sc_backw, sc_index == 0 ? 1 : 0);
                    if (!(*sc_backw)->is_type<scale_grad_weights>())
                        return;

                    auto conv_out_prim = std::make_shared<mutable_data>(prim->id + "_fused_conv_out", memory::attach(dummy_layout, &zero, 1));
                    auto& conv_out_node = p.get_or_create(conv_out_prim);
                    auto conv_out_mem = p.get_engine().allocate_memory(node.get_output_layout());
                    conv_out_node.as<mutable_data>().attach_memory(*conv_out_mem, false);
                    p.add_intermediate(conv_out_node, **bn_backw, 1, true);

                    auto bn_out_prim = std::make_shared<mutable_data>(prim->id + "_fused_bn_out", memory::attach(dummy_layout, &zero, 1));
                    auto& bn_out_node = p.get_or_create(bn_out_prim);
                    auto bn_out_mem = p.get_engine().allocate_memory(bn_node->get_output_layout());
                    bn_out_node.as<mutable_data>().attach_memory(*bn_out_mem, false);
                    p.add_intermediate(bn_out_node, **sc_backw, 0, true);
                }

                auto new_conv = std::make_shared<fused_conv_bn_scale>(prim->id + "_fused", prim->input[0], prim->weights.ref(), prim->bias.ref(), bn_prim->epsilon,
                    scale_prim->input[1], scale_prim->bias, prim->stride, prim->dilation, prim->input_offset, bn_prim->inv_variance,
                    prim->with_activation, prim->activation_negative_slope, prim->output_padding);
                auto& new_node = p.get_or_create(new_conv);
                p.replace(node, new_node);

                while (sc_node->get_dependencies().size() > 1)     // ToDo: here we modify users and dependencies,
                                                                   // It should be done through public methods in program_node/program_impl
                                                                   // to avoid friend declarations
                {
                    auto& dep = sc_node->get_dependency(sc_node->get_dependencies().size() - 1);
                    p.remove_connection(dep, *sc_node);
                    dep.users.push_back(&new_node);
                    if (sc_node->get_dependencies().size() == 1)
                        new_node.dependencies.insert(new_node.dependencies.begin() + 1, &dep);
                    else
                        new_node.dependencies.push_back(&dep);
                }
                p.extract_and_remove(*sc_node);
                while (bn_node->get_dependencies().size() > 1)
                {
                    auto& dep = bn_node->get_dependency(bn_node->get_dependencies().size() - 1);
                    p.remove_connection(dep, *bn_node);
                    new_node.dependencies.push_back(&dep);
                }
                p.extract_and_remove(*bn_node);
                auto inv_var_node = std::find_if(new_node.dependencies.begin(), new_node.dependencies.end(),
                    [&new_conv](const program_node* node) { return node->id().find(new_conv->inv_variance) != std::string::npos; });
                (*inv_var_node)->users.push_back(&new_node);

                if (training)
                {
                    auto user = std::find_if(new_node.get_users().begin(), new_node.get_users().end(),
                        [](const program_node* node) { return node->id().find("_fused_conv_out") != std::string::npos; });
                    p.reverse_connection(new_node, **user);
                    user = std::find_if(new_node.get_users().begin(), new_node.get_users().end(),
                        [](const program_node* node) { return node->id().find("_fused_bn_out") != std::string::npos; });
                    p.reverse_connection(new_node, **user);
                    p.get_processing_order().calculate_BFS_processing_order();
                }
            }
        }
    });
}

void prepare_conv_eltw_fusing::fuse_conv_eltwise(program_impl& p, program_node* node)
{
    // make sure this convolution have only 1 user and it's eltwise
    // maek sure convolution is not an output
    if (node->users.size() != 1 ||
        node->is_output())
        return;

    if (!(*(node->users.begin()))->is_type<eltwise>())
        return;

    convolution_node * conv_node = static_cast<convolution_node*>(node);
    convolution & conv = const_cast<convolution&>(*conv_node->get_primitive());

    // currently works only for this format
    if ( (conv_node->get_output_layout().format != cldnn::format::fs_bs_yx_bsv4_fsv32 || conv_node->get_output_layout().data_type != cldnn::data_types::i8) &&
         (conv_node->get_output_layout().format != cldnn::format::bfyx || conv_node->get_output_layout().data_type != cldnn::data_types::f32) &&
         (conv_node->get_output_layout().format != cldnn::format::yxfb || conv_node->get_output_layout().data_type != cldnn::data_types::f16)
       )
        return;

    auto weights_node_ptr = p.nodes_map.find(conv.weights[0])->second;
    auto filter_size = weights_node_ptr->get_output_layout().size;

    // make sure if this is conv 1x1 its stride is 1x1
    if (filter_size.spatial[0] == 1 && filter_size.spatial[1] == 1)
    {
        if (conv.stride.spatial[0] != 1 || conv.stride.spatial[1] != 1)
            return;
    }
    else
        return;

    eltwise_node * eltw_node = static_cast<eltwise_node*>(*(node->users.begin()));

    // make sure eltwise have only 2 inputs
    // make sure eltwise is not an output
    if (eltw_node->inputs_count() != 2 ||
        eltw_node->is_output())
        return;

    // only single ADD operation is currently supported
    // TODO: enable more
    eltwise & eltw = const_cast<eltwise&>(*eltw_node->get_primitive());
    if (eltw.mode != eltwise_mode::sum)
        return;

    if (eltw_node->get_fused_activation_func() == activation_relu_negative_slope)
    {
        eltw.with_activation = true;
        eltw.activation_negative_slope = eltw_node->get_fused_activation_params().a;
    }
    else
    {
        return;
    }

    int eltw_fused_input_idx; // <-- this input gets fused with eltwise
    int eltw_second_input_idx; // <-- this input is not fused, so we add it in kernel
    // here we check which input gets execute as last one, and fuse it
    if (p.processing_order.get_processing_number(&eltw_node->input(0)) < p.processing_order.get_processing_number(&eltw_node->input(1)))
    {
        eltw_fused_input_idx = 1;
        eltw_second_input_idx = 0;
    }
    else
    {
        eltw_fused_input_idx = 0;
        eltw_second_input_idx = 1;
    }

    // we check if input to fuse is convolution that we're right now processing
    if (eltw_node->input(eltw_fused_input_idx).id() != conv.id)
        return;

    primitive_id conv_id = conv_node->id();

    // get strides for other than our conv input
    std::vector<tensor> new_eltw_strides;
    // conv strides modified by eltwise stride
    tensor new_conv_stride = conv.stride;

    if (eltw.stride.size() == eltw_node->inputs_count())
    {
        // for cases when stride from eltwise must be applied into fused convolution
        new_conv_stride.spatial[0] *= eltw.stride[eltw_fused_input_idx].spatial[0];
        new_conv_stride.spatial[1] *= eltw.stride[eltw_fused_input_idx].spatial[1];
        // stride from non-fused eltwise input
        new_eltw_strides.push_back(eltw.stride[eltw_second_input_idx]);
    }

    auto fused_conv_eltw = std::make_shared<fused_conv_eltwise>(
        conv.id + "_fused_" + eltw.id,
        conv_node->input().id(),
        eltw_node->input(eltw_second_input_idx).id(),
        eltw.mode,
        conv.weights.ref(),
        conv.bias.ref(),
        conv.weights_quantization_factors.ref(),
        conv.output_calibration_factors.ref(),
        conv.input_quantization_factor,
        eltw.output_calibration_factors,
        new_eltw_strides,
        new_conv_stride,
        conv.input_offset,
        conv.dilation,
        conv.with_activation,
        conv.activation_negative_slope,
        eltw.with_activation,
        eltw.activation_negative_slope
        );

    auto& new_node = p.get_or_create(fused_conv_eltw);
    p.replace(*conv_node, new_node);

    // right now new node's user is eltwise, let's clear users and take eltwise's users
    new_node.users.clear();
    p.replace_all_usages(*eltw_node, new_node);

    // TODO: do it better, now it's done in a very ugly way to have good dependency order
    std::vector<program_node*> updated_deps;
    updated_deps.push_back(new_node.dependencies[0]);

    // add second input
    updated_deps.push_back(&eltw_node->input(eltw_second_input_idx));
    eltw_node->input(eltw_second_input_idx).users.push_back(&new_node);

    for (size_t d = 1; d < new_node.dependencies.size(); d++)
    {
        updated_deps.push_back(new_node.dependencies[d]);
    }

    if (eltw_node->output_calibration_term())
    {
        updated_deps.push_back(&eltw_node->output_calibration_factors());
        eltw_node->output_calibration_factors().users.push_back(&new_node);
    }

    new_node.dependencies = updated_deps;

    while (eltw_node->dependencies.size() > 1)
    {
        auto& dep = eltw_node->get_dependency(eltw_node->get_dependencies().size() - 1);
        p.remove_connection(dep, *eltw_node);
    }

    p.extract_and_remove(*eltw_node);
    new_node.recalc_output_layout();
}

void prepare_primitive_fusing::run(program_impl& p)
{
    bool is_debug = p.get_options().get<build_option_type::debug>()->enabled();

    std::list<program_node*> conv_nodes;
    auto itr = p.get_processing_order().begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != p.get_processing_order().end())
    {
        auto node_itr = itr++;
        if ((*node_itr)->is_type<convolution>())
            conv_nodes.push_back(*node_itr);
    }

    // Disabled due to kernel being not optimized
    //itr = conv_nodes.begin();
    //while (itr != conv_nodes.end())
    //{
    //    auto node_itr = itr++;
    //    auto& node = (*node_itr);

    //    fuse_conv_bn_scale(p, node);
    //}

    //This loop tries fusing several reorders one by one (if present) into one reorder
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (node->is_output())
            continue;

        program_helpers::do_for_types<reorder>(*node, [&p, is_debug](reorder_node& node)
        {
            auto& input = node.input();

            //Restrictions:
            // - inputs cannot be padded
            // - primitives input cannot be output
            // - input was optimized
            if (node.has_padded_dependency() || (input.is_output() && !is_debug) || node.get_dependencies().size() != 1 ||
                input.can_be_optimized())
                return;

            // - check if previous node is reorder with 1 user (and if the layouts are the same - remove reorder)
            // - do not fuse if current node has mean subtract
            if (input.get_users().size() != 1 ||
                (!input.is_type<reorder>() && input.get_output_layout() != node.get_users().front()->get_output_layout()) ||
                node.has_mean() || !node.get_primitive()->subtract_per_feature.empty())
                return;

            input.set_output_layout(node.get_output_layout(), false);
            p.extract_and_remove(node);
        });
    }

    itr = p.processing_order.begin();
    while (itr != p.processing_order.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        program_helpers::do_for_types<activation>(*node, [&p, is_debug](activation_node& node)
        {
            auto& input = node.input();

            //Restrictions:
            // - inputs cannot be padded
            // - primitives input cannot be output
            // - no activation additional input
            // - input was optimized
            if (node.has_padded_dependency() || (input.is_output() && !is_debug) || node.is_output() ||
                node.get_dependencies().size() != 1 || input.can_be_optimized())
                return;

            // - check if there is no activation fused already
            // - limit to primitives which implementations support activation fusing
            if (input.get_users().size() != 1 || input.get_fused_activation_func() != activation_none ||
                //TODO: new api needs to be created to read such caps
                //right now use whitelist so no new primitives will be affected in case of lack of fused activation support
                (!input.is_type<batch_norm>() && !input.is_type<concatenation>() && !input.is_type<convolution>() &&
                    !input.is_type<crop>() && !input.is_type<deconvolution>() && !input.is_type<eltwise>() &&
                    !input.is_type<fully_connected>() && !input.is_type<lrn>() && !input.is_type<normalize>() &&
                    !input.is_type<permute>() && !input.is_type<pooling>() && !input.is_type<reorder>() &&
                    !input.is_type<reshape>() && !input.is_type<roi_pooling>() && !input.is_type<scale>() &&
                    !input.is_type<softmax>() && !input.is_type<upsampling>() && !input.is_type<mvn>()))
                return;

            input.set_fused_activation(node.get_primitive()->activation_func, node.get_primitive()->additional_params);
            input.set_output_padding(node.get_output_layout().data_padding);

            p.extract_and_remove(node);
        });
    }

    //This loop tries fusing eltwise (sum) with deconvolution
    itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        fuse_skip_layers(p, node);
    }
}

void prepare_conv_eltw_fusing::run(program_impl& p)
{
    std::list<program_node*> conv_nodes;
    auto itr = p.get_processing_order().begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != p.get_processing_order().end())
    {
        auto node_itr = itr++;
        if ((*node_itr)->is_type<convolution>())
            conv_nodes.push_back(*node_itr);
    }

    //fuse conv + eltwise after activations
    itr = conv_nodes.begin();
    while (itr != conv_nodes.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        fuse_conv_eltwise(p, node);
    }
}

void prepare_conv_eltw_read_write_opt::conv_eltwise_read_write_opt(program_impl& p, program_node* node)
{
    fused_conv_eltwise_node * fused_conv_eltw_node = static_cast<fused_conv_eltwise_node*>(node);
    program_node * second_input_node = &fused_conv_eltw_node->get_dependency(1);
    // output layouts must match
    if (fused_conv_eltw_node->get_output_layout() != second_input_node->get_output_layout()) // check whole layout
    {
        return;
    }
    
    // buffer shared between primitives, if second input is mutable data, then we can reuse this memory
    auto shared_buffer_mem = second_input_node->is_type<mutable_data>() ? second_input_node->as<mutable_data>().get_attached_memory_ptr() : p.get_engine().allocate_memory(node->get_output_layout());

    float zero = 0.0f;
    layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));
    
    // this one is the first one to write data to
    auto rw_output_prim0 = std::make_shared<mutable_data>(fused_conv_eltw_node->id() + "_RW_OPT_use", memory::attach(dummy_layout, &zero, 1));
    // this one already expects data to be inside
    auto rw_output_prim1 = std::make_shared<mutable_data>(fused_conv_eltw_node->id() + "_RW_OPT_reuse", memory::attach(dummy_layout, &zero, 1));

    auto& rw_output_node0 = p.get_or_create(rw_output_prim0);
    auto& rw_output_node1 = p.get_or_create(rw_output_prim1);

    rw_output_node0.as<mutable_data>().attach_memory(*shared_buffer_mem, false);
    rw_output_node1.as<mutable_data>().attach_memory(*shared_buffer_mem, false);

    // add connection between second input node -> rw_output_node0 -> node
    p.add_intermediate(rw_output_node0, *node, 1, true);
    // replace other connections with rw_output_node0
    auto itr = second_input_node->users.begin();
    while (itr != second_input_node->users.end())
    {
        auto& usage = (*itr++);
        if (usage->id() != rw_output_node0.id() && usage->id() != node->id())
        {
            usage->replace_dependency(*second_input_node, rw_output_node0);
        }
    }
    // add connection between node -> rw_output_node1 -> after nodes
    //first find index in our first user's dependency
    size_t dep_idx = 0;
    for (auto dep : (*(node->users.begin()))->dependencies)
    {
        if (dep->id() == node->id())
            break;
        dep_idx++;
    }
    p.add_intermediate(rw_output_node1, **(node->users.begin()), dep_idx, true);
    // replace other connections with rw_output_node1
    itr = node->users.begin();
    while (itr != node->users.end())
    {
        auto& usage = (*itr++);
        if (usage->id() != rw_output_node1.id() && usage->id() != node->id())
        {
            usage->replace_dependency(*node, rw_output_node1);
        }
    }
    fused_conv_eltwise* prim = const_cast<fused_conv_eltwise*>((fused_conv_eltw_node->get_primitive().get()));
    prim->second_input_in_output = true;
}

void prepare_conv_eltw_read_write_opt::run(program_impl& p)
{
    std::list<program_node*> fused_conv_eltw_nodes;
    auto itr = p.get_processing_order().begin(); //note we need to use iterators since currently processed element can be removed
    while (itr != p.get_processing_order().end())
    {
        auto node_itr = itr++;
        if ((*node_itr)->is_type<fused_conv_eltwise>())
            fused_conv_eltw_nodes.push_back(*node_itr);
    }

    //fuse conv + eltwise after activations
    itr = fused_conv_eltw_nodes.begin();
    while (itr != fused_conv_eltw_nodes.end())
    {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        conv_eltwise_read_write_opt(p, node);
    }
}