/*
// Copyright (c) 2019 Intel Corporation
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

#include "pass_manager.h"
#include "program_node.h"

#include "split_inst.h"
#include "convolution_inst.h"
#include "crop_inst.h"
#include "lstm_inst.h"
#include "reshape_inst.h"
#include "upsampling_inst.h"
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
std::string get_id_string(size_t i) {
    std::stringstream ss;
    ss << std::setw(5) << std::setfill('0') << i;
    return ss.str();
}

// ToDo: rewrite methods in this class the same style (maybe: handle_<primitive_name>() ),
//       is it possible to avoid iterating over all nodes several times?
//       do we have any repeated code here, can we make it more readable?
void graph_initializations::replace_nodes(program_impl& p) {
    auto itr = p.nodes_map.begin();
    while (itr != p.nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;

        if (node->is_type<split>()) {
            // check if split is not used by any primitive, as it will be optimized
            if (node->get_users().size() != 0)
                throw std::logic_error("Split layer cannot be used directly! Please use split output \"" + node->id() +
                                       ":<split_output_id>\"!");

            // get_output size and validate split primitive inputs
            auto output_layout = node->get_output_layout();
            auto output_layout_size = output_layout.size;

            auto split_prim = node->as<split>().typed_desc();
            primitive_id input_id = split_prim->input[0];
            auto split_num = split_prim->output_offsets.size();

            std::vector<primitive_id> transformed_ids;

            // create crop for each split ouptut provided
            for (decltype(split_num) i = 0; i < split_num; i++) {
                primitive_id output_id = node->id() + ":" + split_prim->output_ids[i];
                transformed_ids.push_back(output_id);

                auto node_ptr = p.nodes_map.find(output_id)->second;

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
                for (int dimension = 0; dimension < CLDNN_TENSOR_DIM_MAX; dimension++) {
                    reference_input_size.raw[dimension] = (reference_input_size.raw[dimension] == 0)
                                                              ? output_layout_size.raw[dimension]
                                                              : reference_input_size.raw[dimension];
                }

                // update crop primitive
                node_ptr->set_output_padding(output_layout.data_padding);
                auto crop_prim = node_ptr->as<crop>().typed_desc();
                crop_prim->reference_input = reference_input_size;
            }

            // remove input->split connection and remove original split node
            p.remove_connection(node->get_dependency(0), *node);

            p.add_optimized_primitive_info(node->id(), transformed_ids);
            p.optimized_out.push_back(node->id());
            p.nodes_map.erase(node->id());
            continue;
        }

        // find upsampling primitives with bilinear filtering and create deconvolution with proper weights instead
        if (node->is_type<upsampling>()) {
            auto upsampling_prim = node->as<upsampling>().typed_desc();

            if (upsampling_prim->sample_type != upsampling_sample_type::bilinear)
                continue;

            // check if num_filter is not 0 (required for bilinear upsampling)
            if (upsampling_prim->num_filter == 0)
                throw std::logic_error("num_filter in upsampling cannot be 0 in bilinear filtering mode in \"" +
                                       node->id() + "\"!");

            primitive_id upsampling_id = node->id();
            auto& input_node = node->get_dependency(0);

            primitive_id input_id = upsampling_prim->input[0];
            auto num_filter = upsampling_prim->num_filter;

            // setting deconvolution parameters based on upsampling input
            auto scale = static_cast<tensor::value_type>(upsampling_prim->scale);
            tensor stride(1, 1, scale, scale);
            auto offset = static_cast<tensor::value_type>(std::ceil((scale - 1) / 2.f));
            tensor input_offset(0, 0, -offset, -offset);

            // setting weights for deconvolution
            auto kernel_size = static_cast<tensor::value_type>((2 * scale) - (scale % 2));
            layout weights_layout(data_types::f32, format::bfyx, tensor(1, 1, kernel_size, kernel_size));

            std::vector<primitive_id> weights_vec;
            for (uint32_t weights_idx = 0; weights_idx < num_filter; weights_idx++) {
                memory_impl::ptr data_to_allocate = p.get_engine().allocate_memory(weights_layout, 0);
                mem_lock<float> dst{data_to_allocate};
                float* dst_data = dst.data();
                // initialize with bilinear weights data
                auto f = static_cast<uint32_t>(std::ceil(kernel_size / 2.0f));
                float c = (2 * f - 1 - f % 2) / (2.f * f);
                float x = 0.f;
                float y = 0.f;
                for (size_t i = 0; i < weights_layout.count(); ++i) {
                    x = static_cast<float>(i % kernel_size);
                    y = static_cast<float>((i / kernel_size) % kernel_size);
                    dst_data[i] = (1 - std::abs(x / f - c)) * (1 - std::abs(y / f - c));
                }

                // create weights primitive, with dummy memory which will be replaced in firther step
                primitive_id weights_id = upsampling_id + "_deconvolution_weights" + std::to_string(weights_idx);
                layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));
                float zero = 0.f;
                auto weights_prim = std::make_shared<data>(weights_id, memory::attach(dummy_layout, &zero, 1));
                p.get_or_create(weights_prim);

                weights_vec.push_back(weights_id);

                auto weights_node_ptr = p.nodes_map.find(weights_id)->second;

                // attach weights buffer
                auto& data_node = weights_node_ptr->as<data>();
                data_node.attach_memory(*data_to_allocate, false);
            }

            // remove upsampling node, rename it and move to the optimized list
            p.remove_connection(node->get_dependency(0), *node);
            auto rename_id = upsampling_id + "_tmp";
            p.rename(*node, rename_id);

            // create deconvolution primitive
            auto deconv_prim =
                std::make_shared<deconvolution>(upsampling_id, input_id, weights_vec, stride, input_offset);
            p.get_or_create(deconv_prim);

            auto deconv_node_ptr = p.nodes_map.find(upsampling_id)->second;

            auto upsampling_node_ptr = p.nodes_map.find(rename_id)->second;
            p.replace_all_usages(*upsampling_node_ptr, *deconv_node_ptr);
            p.optimized_out.push_back(rename_id);
            p.nodes_map.erase(rename_id);

            // add connections input->deconvolution and weights->deconvolution
            p.add_connection(input_node, *deconv_node_ptr);

            for (uint32_t weights_idx = 0; weights_idx < num_filter; weights_idx++) {
                auto weights_node_ptr = p.nodes_map.find(weights_vec[weights_idx])->second;
                p.add_connection(*weights_node_ptr, *deconv_node_ptr);
            }
            continue;
        }

        // find deconvolution primitives with stride 1 and change them to convolution with trasposed weights
        if (node->is_type<deconvolution>()) {
            if (!p.get_options().get<build_option_type::optimize_data>()->enabled())
                continue;

            auto deconv_prim = node->as<deconvolution>().typed_desc();

            // limit optimization to stride = 1
            if (deconv_prim->stride.spatial[0] != 1 || deconv_prim->stride.spatial[1] != 1 || deconv_prim->gradient())
                continue;

            primitive_id deconv_id = node->id();
            auto& input_node = node->get_dependency(0);

            // disable for 5D
            if (input_node.get_output_layout().format == format::bfzyx)
                continue;

            primitive_id input_id = deconv_prim->input[0];

            // setting convolution parameters based on deconvolution params
            auto stride = deconv_prim->stride;
            auto weights = deconv_prim->weights;
            std::vector<primitive_id> weights_vec;
            for (auto& weights_id : weights) weights_vec.push_back(weights_id);
            auto biases = deconv_prim->bias;
            std::vector<primitive_id> bias_vec;
            for (auto& bias_id : biases) bias_vec.push_back(bias_id);
            auto input_offset = deconv_prim->input_offset;
            auto with_activation = deconv_prim->with_activation;
            auto activation_negative_slope = deconv_prim->activation_negative_slope;
            auto output_padding = deconv_prim->output_padding;

            // remove deconvolution node and its connections to weights and biases, rename it and move to the optimized
            // list
            tensor filter_size = {1, 1, 1, 1, 1};
            p.remove_connection(node->get_dependency(0), *node);
            for (auto& weights_id : weights_vec) {
                auto weights_node_ptr = p.nodes_map.find(weights_id)->second;
                p.remove_connection(*weights_node_ptr, *node);
                // get filter spatial sizes for input offset adjustment, perform this only once as all filters shouls
                // have same size
                if (weights_id == weights_vec[0])
                    filter_size = weights_node_ptr->get_output_layout().size;
            }

            input_offset.spatial[0] = std::abs(input_offset.spatial[0]) - (filter_size.spatial[0] - 1);
            input_offset.spatial[1] = std::abs(input_offset.spatial[1]) - (filter_size.spatial[1] - 1);
            input_offset.spatial[2] = std::abs(input_offset.spatial[2]) - (filter_size.spatial[2] - 1);

            if (!bias_vec.empty()) {
                for (auto& bias_id : bias_vec) {
                    auto bias_id_node_ptr = p.nodes_map.find(bias_id)->second;
                    p.remove_connection(*bias_id_node_ptr, *node);
                }
            }
            auto rename_id = deconv_id + "_tmp";
            p.rename(*node, rename_id);

            // create convolution primitive
            if (biases.size() != 0) {
                auto conv_prim = std::make_shared<convolution>(deconv_id,
                                                               input_id,
                                                               weights_vec,
                                                               bias_vec,
                                                               stride,
                                                               input_offset,
                                                               tensor{1, 1, 1, 1},
                                                               with_activation,
                                                               activation_negative_slope,
                                                               output_padding);
                p.get_or_create(conv_prim);
            } else {
                auto conv_prim = std::make_shared<convolution>(deconv_id,
                                                               input_id,
                                                               weights_vec,
                                                               stride,
                                                               input_offset,
                                                               tensor{1, 1, 1, 1},
                                                               with_activation,
                                                               activation_negative_slope,
                                                               output_padding);
                p.get_or_create(conv_prim);
            }

            auto conv_node_ptr = p.nodes_map.find(deconv_id)->second;
            auto conv_node = &conv_node_ptr->as<convolution>();
            conv_node->set_transposed(true);

            // add connections input->convolution, weights->convolution and bias->convolution
            p.add_connection(input_node, *conv_node_ptr);

            for (auto& weights_id : weights_vec) {
                auto weights_node_ptr = p.nodes_map.find(weights_id)->second;
                p.add_connection(*weights_node_ptr, *conv_node_ptr);
            }

            if (!bias_vec.empty()) {
                for (auto& bias_id : bias_vec) {
                    auto bias_id_node_ptr = p.nodes_map.find(bias_id)->second;
                    p.add_connection(*bias_id_node_ptr, *conv_node_ptr);
                }
            }

            auto deconv_node_ptr = p.nodes_map.find(rename_id)->second;
            p.replace_all_usages(*deconv_node_ptr, *conv_node_ptr);
            p.optimized_out.push_back(rename_id);
            p.nodes_map.erase(rename_id);

            continue;
        }
    }
}

void graph_initializations::handle_detection_output(program_impl& p) {
    auto itr = p.nodes_map.begin();  // note we need to use iterators since currently processed element can be removed
    while (itr != p.nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = *(*node_itr).second;
        // Create second part detection output primitive and replace nodes names - do it only once
        if ((p.get_options().get<build_option_type::detection_output_gpu>()->enabled()) &&
            (node.is_type<detection_output>()) &&
            (node.id().find("_pre") ==
             std::string::npos)) {  // ToDo: this will fail if user will name the primitive with using _pre like do_pre
                                    //       we need to use node mark() or some other idea to prevent it
            // rename detection output
            const primitive_id detect_out_node_name = node.id();
            const primitive_id new_primitive_id = detect_out_node_name + "_pre";
            p.rename(node, new_primitive_id);

            auto detect_out_prim = node.as<detection_output>().typed_desc();
            // Create new primitive, "keep top k" part of detection output
            // ToDo: add a default parameters to the detection_output_sort class constructor to get rid off this
            // initialization from here
            auto detect_out_sort_prim =
                std::make_shared<detection_output_sort>(detect_out_node_name,
                                                        node.id(),
                                                        // not important params here - it will be set during
                                                        // "primitive_impl* create" func in "detection_output_sort_gpu"
                                                        0,      // num_images
                                                        0,      // num_classes
                                                        0,      // keep_top_k
                                                        false,  // share_location
                                                        0,      // top_k
                                                        -1,     // background_label_id
                                                        detect_out_prim->output_padding);

            p.get_or_create(detect_out_sort_prim);

            auto sort_node = p.nodes_map.find(detect_out_node_name)->second;

            // Add connection to second part of detection output
            if (node.get_users().size()) {
                p.add_intermediate(*sort_node, *(node.get_users().front()), 0, false);
            } else {
                p.add_connection(node, *sort_node);
            }
        }
    }
}

void graph_initializations::handle_lstm(program_impl& p) {
    bool has_lstm_children;
    auto itr = p.nodes_map.begin();  // note we need to use iterators since currently processed element can be removed
    while (itr != p.nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;
        has_lstm_children = false;
        // replace lstm node with lstm_gemm and lstm_elt nodes
        if (node->is_type<lstm>()) {
            bool initial_hidden_term = node->as<lstm>().initial_hidden_term();
            bool initial_cell_term = node->as<lstm>().initial_cell_term();
            bool bias_term = node->as<lstm>().bias_term();
            auto lstm_prim = node->as<lstm>().typed_desc();
            primitive_id weights_id = lstm_prim->weights;
            primitive_id recurrent_id = lstm_prim->recurrent;
            primitive_id bias_id = bias_term ? lstm_prim->bias : "";
            primitive_id initial_hidden_id = initial_hidden_term ? lstm_prim->initial_hidden : "";
            primitive_id initial_cell_id = initial_cell_term ? lstm_prim->initial_cell : "";

            // removing connection with weights to get proper dependency order for next operations
            p.remove_connection(*p.nodes_map.at(weights_id), *node);
            p.remove_connection(*p.nodes_map.at(recurrent_id), *node);
            if (bias_term)
                p.remove_connection(*p.nodes_map.at(bias_id), *node);
            if (initial_hidden_term)
                p.remove_connection(*p.nodes_map.at(initial_hidden_id), *node);
            if (initial_cell_term)
                p.remove_connection(*p.nodes_map.at(initial_cell_id), *node);

            // calculating sizes
            auto input_size = node->get_dependency(0).get_output_layout().size;
            auto recurrent_size = p.nodes_map.at(recurrent_id)->get_output_layout().size;

            // hidden tensor size = [batch, seq, hidden_size, direction]
            // the output of the element wise operation is cropped and used in the next time step
            // sequence_len = 1 and direction = 1. The backward pass is separated from the forward pass
            auto hidden_size = tensor(input_size.batch[0], 1, recurrent_size.spatial[0], 1);

            size_t directions = recurrent_size.feature[0];
            size_t input_directions = input_size.spatial[1];
            size_t num_input_dependencies = node->get_dependencies().size();
            size_t input_vector_size = node->as<lstm>().sequence_len();
            size_t sequence_len = input_vector_size;

            // Calculate the input sequence length for the lstm node
            // Case 1: If the input comes in as a concatenated input i.e. the
            // input is not divided into sequence elements
            if (input_vector_size == 1 && num_input_dependencies == 1) {
                // Either the input actually has 1 sequence element
                auto& input = node->get_dependency(0);
                auto input_layout = input.get_output_layout();
                tensor input_tensor = input_layout.size;

                // Get the sequence length from the input to LSTM
                sequence_len = input_layout.size.feature[0];

                // If the input's feature/sequence length field is > 1, i.e. If
                // the sequence elements are concatenated into one single input
                // then it has to be split into individual sequence elements
                if (sequence_len > 1) {
                    for (size_t sequence_element = 0; sequence_element < sequence_len; sequence_element++) {
                        primitive_id crop_id = input.id() + ":crop:" + get_id_string(sequence_element);
                        tensor crop_tensor{input_tensor.batch[0], 1, input_tensor.spatial[0], input_tensor.spatial[1]};
                        tensor offset_tensor{0, static_cast<tensor::value_type>(sequence_element), 0, 0};
                        auto input_crop = std::make_shared<crop>(crop_id, input.id(), crop_tensor, offset_tensor);
                        auto& input_crop_node = p.get_or_create(input_crop);

                        // Add the crop nodes as user for input
                        p.add_connection(node->get_dependency(0), input_crop_node);

                        // Connect crop with lstm
                        p.add_connection(input_crop_node, *node);
                    }

                    // We have the sequence elements (cropped inputs) as input to LSTM.
                    // The original input is no longer a dependency to LSTM.
                    // Remove the input node as a dependency to LSTM
                    p.remove_connection(node->get_dependency(0), *node);

                    // Update the total no. of input dependecies
                    num_input_dependencies = node->get_dependencies().size();
                }
            // if the sequence has a single element but it has multiple inputs then
            // the parent of this lstm is an lstm node. If this is a bidirectional lstm
            // then the sequence length is the number of dependencies divided by 2.
            } else if (input_vector_size == 1 && num_input_dependencies > 1) {
                sequence_len = (directions == 1) ? num_input_dependencies : num_input_dependencies / 2;
            }

            // check if this lstm node has an lstm child
            for (auto& user : node->get_users()) {
                if (user->is_type<lstm>()) {
                    has_lstm_children = true;
                }
            }

            bool emit_last_cell = lstm_prim->output_selection == cldnn_lstm_output_hidden_cell ||
                                  lstm_prim->output_selection == cldnn_lstm_output_sequence_cell;
            bool emit_sequence = lstm_prim->output_selection == cldnn_lstm_output_sequence_cell ||
                                 lstm_prim->output_selection == cldnn_lstm_output_sequence;

            std::vector<program_node*> cell_list(directions * sequence_len);
            std::vector<program_node*> hidden_list(directions * sequence_len);
            std::map<size_t, std::pair<primitive_id, program_node*>> output_map;
            auto dependencies = node->get_dependencies();

            // lstm expanding
            for (size_t dir = 0; dir < directions; ++dir) {
                auto hidden_id = initial_hidden_id;
                auto cell_id = initial_cell_id;
                for (size_t i = 0; i < sequence_len; ++i) {
                    size_t idx = i + dir * sequence_len;
                    primitive_id lstm_gemm_id = node->id() + ":lstm_gemm" + get_id_string(idx);
                    primitive_id lstm_elt_id = node->id() + ":lstm_elt" + get_id_string(idx);
                    primitive_id crop_id = node->id() + ":crop" + get_id_string(idx);

                    size_t input_idx = i;
                    // for bidirectional lstms, if first LSTM layer then reverse input
                    // for subsequent stacked layers the input is strided on the dir dimension
                    if (directions > 0) {
                        if (num_input_dependencies > sequence_len) {  // stacked layer
                            input_idx = dir * sequence_len + i;
                        } else {
                            if ((input_directions < 2) && dir > 0) {  // first layer
                                input_idx = sequence_len - i - 1;
                            }
                        }
                    }

                    // primitive_id lstm_gemm_input_id = node->get_dependency(input_idx).get_primitive()->id;
                    // the line below requires an attention: get_org_primitive_id() might not be an actual id of a node
                    // (see rename method) ToDO: ensure that get_org_primitive_id() is suitable here
                    primitive_id lstm_gemm_input_id = node->get_dependency(input_idx).get_org_primitive_id();

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
                    p.add_connection(node->get_dependency(input_idx), n1);
                    // adding weights and initial values to lstm_gemm
                    p.add_connection(*p.nodes_map.at(weights_id), n1);
                    p.add_connection(*p.nodes_map.at(recurrent_id), n1);
                    if (bias_term)
                        p.add_connection(*p.nodes_map.at(bias_id), n1);

                    // adding cell and hiddens as dependencies
                    if (i > 0) {
                        p.add_connection(*cell_list[size_t(i - 1) * directions + dir], n2);
                        p.add_connection(*hidden_list[size_t(i - 1) * directions + dir], n1);
                    } else {  // if initial values are present
                        if (initial_hidden_term)
                            p.add_connection(*p.nodes_map.at(hidden_id), n1);
                        if (initial_cell_term)
                            p.add_connection(*p.nodes_map.at(cell_id), n2);
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
                            for (auto& user : node->get_users()) {
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
                std::vector<primitive_id> output_ids_offsets;
                for (auto& e : output_map) {
                    output_ids_offsets.push_back(e.second.first);
                }
                primitive_id original_id = node->id();
                primitive_id concatenation_id = original_id + ":concat";
                auto concatenation_primitive =
                    std::make_shared<concatenation>(concatenation_id, output_ids_offsets, concatenation::along_f);
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

                    tensor output_size{input_size.batch[0],
                                       static_cast<int32_t>(concatenate_len),
                                       hidden_size.spatial[0],
                                       (int32_t)directions};
                    primitive_id reshape_id = original_id + ":reshape";
                    auto reshape_primitive = std::make_shared<reshape>(reshape_id, concatenation_id, output_size);
                    auto& reshape_node = p.get_or_create(reshape_primitive);
                    p.add_connection(concatenation_node, reshape_node);
                    p.replace_all_usages(*node, reshape_node);
                } else {
                    p.replace_all_usages(*node, concatenation_node);
                }
            }
            // removing expanded node
            p.remove_all_connections(*node);
            p.nodes_map.erase(node->id());
            continue;
        }
    }
}

void graph_initializations::handle_dynamic_lstm(program_impl& p) {
    /*
        We need to replace lstm_dynamic with lstm_dynamic_input + lstm_dynamic_timeloop.
    */
    auto itr = p.nodes_map.begin();
    while (itr != p.nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;

        if (node->is_type<lstm_dynamic>()) {
            // [0] Prepare helper temp variables.
            auto& lstm_dynamic_node = node->as<lstm_dynamic>();
            auto& node_id = lstm_dynamic_node.id();
            auto input_id = node->get_primitive()->input.at(0);
            auto dyn_length_id = lstm_dynamic_node.dyn_length_id();
            auto weights_id = lstm_dynamic_node.weights_id();
            auto recurrent_id = lstm_dynamic_node.recurrent_id();
            auto bias_id = lstm_dynamic_node.bias_id();
            auto init_hidden_id = lstm_dynamic_node.initial_hidden_id();
            auto init_cell_id = lstm_dynamic_node.initial_cell_id();
            auto last_hidden_id = lstm_dynamic_node.last_hidden_state_id();
            auto last_cell_id = lstm_dynamic_node.last_cell_state_id();
            auto clip = lstm_dynamic_node.clip();
            auto input_forget = lstm_dynamic_node.input_forget();
            std::string suffix = "__cldnn_";

            // [1] Add lstm_dynamic_input
            auto lstm_dynamic_input_primitive =
                std::make_shared<lstm_dynamic_input>(node_id + suffix + "input",
                                                     input_id,
                                                     dyn_length_id,
                                                     weights_id,
                                                     bias_id,
                                                     node->get_primitive()->output_padding);
            auto& lstm_dynamic_input_node = p.get_or_create(lstm_dynamic_input_primitive);
            p.add_connection(node->get_dependency(0), lstm_dynamic_input_node);  // connect real input to dlstm_input
            // connect other deps
            p.add_connection(p.get_node(dyn_length_id), lstm_dynamic_input_node);
            p.add_connection(p.get_node(weights_id), lstm_dynamic_input_node);
            if (!bias_id.empty())
                p.add_connection(p.get_node(bias_id), lstm_dynamic_input_node);
            lstm_dynamic_input_node.get_output_layout();  // calc out layout

            auto lstm_dynamic_timeloop_primitive =
                std::make_shared<lstm_dynamic_timeloop>(node_id + suffix + "timeloop",
                                                        lstm_dynamic_input_node.id(),
                                                        dyn_length_id,
                                                        recurrent_id,
                                                        last_hidden_id,
                                                        last_cell_id,
                                                        init_hidden_id,
                                                        init_cell_id,
                                                        clip,
                                                        input_forget,
                                                        lstm_dynamic_input_primitive->output_padding);
            auto& lstm_dynamic_timeloop_node = p.get_or_create(lstm_dynamic_timeloop_primitive);
            p.add_connection(lstm_dynamic_input_node,
                             lstm_dynamic_timeloop_node);  // connect dlstm_input to dlstm_timeloop
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
            p.replace_all_usages(*node, lstm_dynamic_timeloop_node);
            p.remove_all_connections(*node);
            p.remove_if_dangling(*node);
            p.rename(lstm_dynamic_timeloop_node, node_id);  // get original id

            // we dont have to set output since it will be done in next graph_opts step
        }
    }
}

void graph_initializations::set_outputs(program_impl& p) {
    auto outputs_option = p.get_options().get<build_option_type::outputs>();
    if (!outputs_option->outputs.empty()) {
        for (auto const& output : outputs_option->outputs) {
            auto o_node = p.nodes_map.at(output);
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

void graph_initializations::run(program_impl& p) {
    replace_nodes(p);
    handle_detection_output(p);
    handle_lstm(p);
    handle_dynamic_lstm(p);
    set_outputs(p);
    p.get_processing_order().calc_processing_order(p);
}
}  // namespace cldnn
