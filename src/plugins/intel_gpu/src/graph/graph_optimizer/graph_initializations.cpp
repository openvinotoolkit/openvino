// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"

#include "split_inst.h"
#include "convolution_inst.h"
#include "crop_inst.h"
#include "lstm_inst.h"
#include "reshape_inst.h"
#include "resample_inst.h"
#include "depth_to_space_inst.h"
#include "lstm_dynamic_inst.h"
#include "lstm_dynamic_input_inst.h"
#include "lstm_dynamic_timeloop_inst.h"
#include "mutable_data_inst.h"
#include "arg_max_min_inst.h"

#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>

using namespace cldnn;

namespace cldnn {
namespace {
std::string get_id_string(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}
}  // namespace

void graph_initializations::handle_split_node(program& p, split_node& node) {
    if (!node.get_users().empty()) {
        throw std::logic_error("Split layer cannot be used directly! Please use split output \"" + node.id() +
                               ":<split_output_id>\"!");
    }
    // get_output size and validate split primitive inputs
    layout output_layout = node.get_output_layout();
    tensor output_layout_size = output_layout.get_tensor();

    auto split_prim = node.typed_desc();
    std::size_t split_num = split_prim->output_offsets.size();

    std::vector<primitive_id> transformed_ids;

    // create crop for each split output provided
    for (std::size_t i = 0; i < split_num; i++) {
        primitive_id output_id = node.id() + ":" + split_prim->output_ids[i];

        auto output_node_itr = p.nodes_map.find(output_id);
        if (output_node_itr == p.nodes_map.end()) {
            continue;
        }

        transformed_ids.push_back(std::move(output_id));

        auto node_ptr = output_node_itr->second;

        // calculate crop reference input size
        tensor reference_input_size;

        // For all the split offsets before the last split offset, the size can be calculated
        // size_of_offset[n] = offset[n + 1] - offset[n];
        if (i != (split_num - 1)) {
            reference_input_size += split_prim->output_offsets[i + 1] - split_prim->output_offsets[i];
        } else {  // For the last split i.e. size[split_num - 1] = split_input.size - offsets[n];
            reference_input_size += output_layout_size - split_prim->output_offsets[i];
        }

        // For all the other dimensions, copy from the split_input
        for (int32_t dimension = 0; dimension < tensor_dim_max; dimension++) {
            if (reference_input_size.raw[dimension] == 0) {
                reference_input_size.raw[dimension] = output_layout_size.raw[dimension];
            }
        }

        // update crop primitive
        node_ptr->set_output_padding(output_layout.data_padding);
        auto crop_prim = node_ptr->as<crop>().typed_desc();
        crop_prim->reference_input = reference_input_size;
    }

    // remove input->split connection and remove original split node
    p.remove_connection(node.input(), node);

    p.add_optimized_primitive_info(node.id(), transformed_ids);
    p.optimized_out.push_back(node.id());
    p.nodes_map.erase(node.id());
}

void graph_initializations::handle_lstm_node(program& p, lstm_node& node) {
    // lstm_node& lstm_node = node->as<lstm>();
    bool initial_hidden_term = node.initial_hidden_term();
    bool initial_cell_term = node.initial_cell_term();
    bool bias_term = node.bias_term();
    auto lstm_prim = node.typed_desc();
    primitive_id weights_id = lstm_prim->weights;
    primitive_id recurrent_id = lstm_prim->recurrent;
    primitive_id bias_id = bias_term ? lstm_prim->bias : "";
    primitive_id initial_hidden_id = initial_hidden_term ? lstm_prim->initial_hidden : "";
    primitive_id initial_cell_id = initial_cell_term ? lstm_prim->initial_cell : "";

    // removing connection with weights to get proper dependency order for next operations
    p.remove_connection(p.get_node(weights_id), node);
    p.remove_connection(p.get_node(recurrent_id), node);
    if (bias_term)
        p.remove_connection(p.get_node(bias_id), node);
    if (initial_hidden_term)
        p.remove_connection(p.get_node(initial_hidden_id), node);
    if (initial_cell_term)
        p.remove_connection(p.get_node(initial_cell_id), node);

    // calculating sizes
    program_node& input = node.input();
    layout input_layout = input.get_output_layout();
    tensor recurrent_size = p.get_node(recurrent_id).get_output_layout().get_tensor();

    // hidden tensor size = [batch, seq, hidden_size, direction]
    // the output of the element wise operation is cropped and used in the next time step
    // sequence_len = 1 and direction = 1. The backward pass is separated from the forward pass
    auto hidden_size = tensor(input_layout.batch(), 1, recurrent_size.spatial[0], 1);

    size_t directions = recurrent_size.feature[0];
    size_t num_input_dependencies = node.get_dependencies().size();
    size_t sequence_len = node.sequence_len();

    // Calculate the input sequence length for the lstm node
    // Case 1: If the input comes in as a concatenated input i.e. the
    // input is not divided into sequence elements
    if (sequence_len == 1 && num_input_dependencies == 1) {
        // Get the sequence length from the input to LSTM
        sequence_len = input_layout.feature();

        // If the input's feature/sequence length field is > 1, i.e. If
        // the sequence elements are concatenated into one single input
        // then it has to be split into individual sequence elements
        if (sequence_len > 1) {
            for (size_t sequence_element = 0; sequence_element < sequence_len; sequence_element++) {
                primitive_id crop_id = input.id() + ":crop:" + get_id_string(sequence_element);
                tensor crop_tensor{input_layout.batch(), 1, input_layout.spatial(0), input_layout.spatial(1)};
                tensor offset_tensor{0, static_cast<tensor::value_type>(sequence_element), 0, 0};
                auto input_crop = std::make_shared<crop>(crop_id, input.id(), crop_tensor, offset_tensor);
                auto& input_crop_node = p.get_or_create(input_crop);

                // Add the crop nodes as user for input
                p.add_connection(input, input_crop_node);

                // Connect crop with lstm
                p.add_connection(input_crop_node, node);
            }

            // We have the sequence elements (cropped inputs) as input to LSTM.
            // The original input is no longer a dependency to LSTM.
            // Remove the input node as a dependency to LSTM
            p.remove_connection(input, node);

            // Update the total no. of input dependecies
            num_input_dependencies = node.get_dependencies().size();
        }
    // if the sequence has a single element but it has multiple inputs then
    // the parent of this lstm is an lstm node. If this is a bidirectional lstm
    // then the sequence length is the number of dependencies divided by 2.
    } else if (sequence_len == 1 && num_input_dependencies > 1) {
        sequence_len = (directions == 1) ? num_input_dependencies : num_input_dependencies / 2;
    }

    // check if this lstm node has an lstm child
    bool has_lstm_children = false;
    for (auto& user : node.get_users()) {
        if (user->is_type<lstm>()) {
            has_lstm_children = true;
        }
    }

    bool emit_last_cell = lstm_prim->output_selection == lstm_output_selection::hidden_cell ||
                            lstm_prim->output_selection == lstm_output_selection::sequence_cell;
    bool emit_sequence = lstm_prim->output_selection == lstm_output_selection::sequence_cell ||
                            lstm_prim->output_selection == lstm_output_selection::sequence;

    std::vector<program_node*> cell_list(directions * sequence_len);
    std::vector<program_node*> hidden_list(directions * sequence_len);
    std::map<size_t, std::pair<primitive_id, program_node*>> output_map;
    size_t input_directions = input_layout.spatial(1);

    // lstm expanding
    for (size_t dir = 0; dir < directions; ++dir) {
        auto hidden_id = initial_hidden_id;
        auto cell_id = initial_cell_id;
        for (size_t i = 0; i < sequence_len; ++i) {
            size_t idx = i + dir * sequence_len;
            primitive_id lstm_gemm_id = node.id() + ":lstm_gemm" + get_id_string(idx);
            primitive_id lstm_elt_id = node.id() + ":lstm_elt" + get_id_string(idx);
            primitive_id crop_id = node.id() + ":crop" + get_id_string(idx);

            size_t input_idx = i;
            // for bidirectional lstms, if first LSTM layer then reverse input
            // for subsequent stacked layers the input is strided on the dir dimension
            if (num_input_dependencies > sequence_len) {  // stacked layer
                input_idx = dir * sequence_len + i;
            } else if ((input_directions < 2) && dir > 0) {  // first layer
                input_idx = sequence_len - i - 1;
            }

            // primitive_id lstm_gemm_input_id = node->get_dependency(input_idx).get_primitive()->id;
            // the line below requires an attention: get_org_primitive_id() might not be an actual id of a node
            // (see rename method) ToDO: ensure that get_org_primitive_id() is suitable here
            primitive_id lstm_gemm_input_id = node.get_dependency(input_idx).get_org_primitive_id();

            auto lstm_gemm_node = std::make_shared<lstm_gemm>(lstm_gemm_id,
                                                              lstm_gemm_input_id,
                                                              weights_id,
                                                              recurrent_id,
                                                              bias_id,
                                                              hidden_id,
                                                              (uint32_t)dir);
            auto& n1 = p.get_or_create(lstm_gemm_node);

            auto lstm_elt_node = std::make_shared<lstm_elt>(lstm_elt_id,
                                                            lstm_gemm_id,
                                                            cell_id,
                                                            lstm_prim->clip,
                                                            lstm_prim->input_forget,
                                                            lstm_prim->activations,
                                                            lstm_prim->activation_params,
                                                            lstm_prim->offset_order,
                                                            (uint32_t)dir);
            auto& n2 = p.get_or_create(lstm_elt_node);
            // adding lstm_elt as user
            p.add_connection(n1, n2);
            // adding dependecy to lstm_gemm node
            // input
            p.add_connection(node.get_dependency(input_idx), n1);
            // adding weights and initial values to lstm_gemm
            p.add_connection(p.get_node(weights_id), n1);
            p.add_connection(p.get_node(recurrent_id), n1);
            if (bias_term)
                p.add_connection(p.get_node(bias_id), n1);

            // adding cell and hiddens as dependencies
            if (i > 0) {
                p.add_connection(*cell_list[(i - 1) * directions + dir], n2);
                p.add_connection(*hidden_list[(i - 1) * directions + dir], n1);
            } else {  // if initial values are present
                if (initial_hidden_term)
                    p.add_connection(p.get_node(hidden_id), n1);
                if (initial_cell_term)
                    p.add_connection(p.get_node(cell_id), n2);
            }

            // lstm_hidden
            {
                hidden_id = crop_id + ":hidden";
                auto crop_hidden =
                    std::make_shared<crop>(hidden_id, lstm_elt_id, hidden_size, tensor{0, 0, 0, 0});
                auto& n3 = p.get_or_create(crop_hidden);
                // adding eltwise as dependency to hidden
                p.add_connection(n2, n3);

                // if parent is lstm adding hiddens as dependency
                if (has_lstm_children) {
                    for (auto& user : node.get_users()) {
                        p.add_connection(n3, *user);
                    }
                }
                hidden_list[i * directions + dir] = &n3;
                if (i == sequence_len - 1 || emit_sequence) {
                    output_map[i * directions + dir] = {hidden_id, &n3};
                }
            }

            // lstm_cell
            if (i < sequence_len - 1 || emit_last_cell) {
                cell_id = crop_id + ":cell";
                auto crop_cell = std::make_shared<crop>(cell_id, lstm_elt_id, hidden_size, tensor{0, 1, 0, 0});
                auto& n4 = p.get_or_create(crop_cell);
                p.add_connection(n2, n4);
                cell_list[i * directions + dir] = &n4;
                if (i == sequence_len - 1) {
                    output_map[sequence_len * directions + dir] = {cell_id, &n4};
                }
            }
        }
    }
    // if there is no next lstm, concatenation is created
    if (!has_lstm_children) {
        std::vector<input_info> output_ids_offsets;
        for (auto& e : output_map) {
            output_ids_offsets.push_back(input_info(e.second.first));
        }
        primitive_id concatenation_id = node.id() + ":concat";
        auto concatenation_primitive = std::make_shared<concatenation>(concatenation_id, output_ids_offsets, 1);
        auto& concatenation_node = p.get_or_create(concatenation_primitive);
        for (auto& e : output_map) {
            p.add_connection(*e.second.second, concatenation_node);
        }
        if (directions == 2) {
            // bidirectional support requires concatenations along the direction and sequence axis
            // instead we can concatenate along the sequence axis and reshape the tensor to the account
            // for the direction
            size_t concatenate_len = emit_sequence ? sequence_len : 1;
            if (emit_last_cell)
                concatenate_len++;

            tensor output_size{input_layout.batch(),
                               static_cast<int32_t>(concatenate_len),
                               hidden_size.spatial[0],
                               (int32_t)directions};
            auto reshape_primitive = std::make_shared<reshape>(node.id() + ":reshape", concatenation_id, output_size);
            auto& reshape_node = p.get_or_create(reshape_primitive);
            p.add_connection(concatenation_node, reshape_node);
            p.replace_all_usages(node, reshape_node);
        } else {
            p.replace_all_usages(node, concatenation_node);
        }
    }
    // removing expanded node
    p.remove_all_connections(node);
    p.nodes_map.erase(node.id());
}

void graph_initializations::handle_dynamic_lstm_node(program& p, lstm_dynamic_node& node) {
    // [0] Prepare helper temp variables.
    // auto& lstm_dynamic_node = node->as<lstm_dynamic>();
    auto& node_id = node.id();
    auto input_id = node.get_primitive()->input.at(0);
    auto dyn_length_id = node.dyn_length_id();
    auto weights_id = node.weights_id();
    auto bias_id = node.bias_id();
    std::string suffix = "__cldnn_";

    // [1] Add lstm_dynamic_input
    auto lstm_dynamic_input_primitive =
        std::make_shared<lstm_dynamic_input>(node_id + suffix + "input",
                                             input_id,
                                             dyn_length_id,
                                             weights_id,
                                             bias_id,
                                             node.get_primitive()->output_paddings[0]);
    auto& lstm_dynamic_input_node = p.get_or_create(lstm_dynamic_input_primitive);
    p.add_connection(node.input(), lstm_dynamic_input_node);  // connect real input to dlstm_input
    // connect other deps
    p.add_connection(p.get_node(dyn_length_id), lstm_dynamic_input_node);
    p.add_connection(p.get_node(weights_id), lstm_dynamic_input_node);
    if (!bias_id.empty())
        p.add_connection(p.get_node(bias_id), lstm_dynamic_input_node);
    lstm_dynamic_input_node.get_output_layout();  // calc out layout

    auto recurrent_id = node.recurrent_id();
    auto init_hidden_id = node.initial_hidden_id();
    auto init_cell_id = node.initial_cell_id();
    auto last_hidden_id = node.last_hidden_state_id();
    auto last_cell_id = node.last_cell_state_id();
    auto lstm_dynamic_timeloop_primitive =
        std::make_shared<lstm_dynamic_timeloop>(node_id + suffix + "timeloop",
                                                lstm_dynamic_input_node.id(),
                                                dyn_length_id,
                                                recurrent_id,
                                                last_hidden_id,
                                                last_cell_id,
                                                init_hidden_id,
                                                init_cell_id,
                                                node.clip(),
                                                node.input_forget(),
                                                lstm_dynamic_input_primitive->output_paddings[0]);
    auto& lstm_dynamic_timeloop_node = p.get_or_create(lstm_dynamic_timeloop_primitive);
    p.add_connection(lstm_dynamic_input_node, lstm_dynamic_timeloop_node);  // connect dlstm_input to dlstm_timeloop
    // connect other deps
    p.add_connection(p.get_node(dyn_length_id), lstm_dynamic_timeloop_node);
    p.add_connection(p.get_node(recurrent_id), lstm_dynamic_timeloop_node);

    // [hack] reversed dependecies so the prociessing/execution order will be valid (from the user persepctive)
    // It means that this optional outputs for sure will be "executed" layer.
    // This connection will be reversed (to normal state) later in program.cpp (right after caluticaiton prcoessing order)!
    if (!last_hidden_id.empty())
        p.add_connection(lstm_dynamic_timeloop_node, p.get_node(last_hidden_id));
    if (!last_cell_id.empty())
        p.add_connection(lstm_dynamic_timeloop_node, p.get_node(last_cell_id));
    // [hack end]
    if (!init_hidden_id.empty())
        p.add_connection(p.get_node(init_hidden_id), lstm_dynamic_timeloop_node);
    if (!init_cell_id.empty())
        p.add_connection(p.get_node(init_cell_id), lstm_dynamic_timeloop_node);
    lstm_dynamic_timeloop_node.get_output_layout();  // calc out layout

    // [2] Finally replace original node with the new ones.
    p.replace_all_usages(node, lstm_dynamic_timeloop_node);
    p.remove_all_connections(node);
    p.remove_if_dangling(node);
    p.rename(lstm_dynamic_timeloop_node, node_id);  // get original id

    // we dont have to set output since it will be done in next graph_opts step
}

void graph_initializations::set_outputs(program& p) {
    auto custom_outputs = p.get_config().get_property(ov::intel_gpu::custom_outputs);
    if (!custom_outputs.empty()) {
        for (auto const& output : custom_outputs) {
            auto o_node = p.get_node_ptr(output);
            o_node->set_output(true);
            p.outputs.push_back(o_node.get());
        }
    } else {
        for (auto& node : p.nodes_map)
            if (node.second->is_endpoint()) {
                node.second->set_output(true);
                p.outputs.push_back(node.second.get());
            }
    }
}

void graph_initializations::run(program& p) {
    auto itr = p.nodes_map.begin();
    while (itr != p.nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = node_itr->second;
        if (node->is_type<split>()) {
            handle_split_node(p, node->as<split>());
        } else if (node->is_type<lstm>()) {
            handle_lstm_node(p, node->as<lstm>());
        } else if (node->is_type<lstm_dynamic>()) {
            handle_dynamic_lstm_node(p, node->as<lstm_dynamic>());
        }
    }
    set_outputs(p);
    p.get_processing_order().calc_processing_order(p);
}
}  // namespace cldnn
