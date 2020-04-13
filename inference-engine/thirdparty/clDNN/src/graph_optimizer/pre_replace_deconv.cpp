/*
// Copyright (c) 2018-2019 Intel Corporation
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

#include "program_helpers.h"
#include "pass_manager.h"

#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include "error_handler.h"

void pre_replace_deconv::run(program_impl& p) {
    bool update_processing_order = false;
    auto itr = p.nodes_map.begin();
    while (itr != p.nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;
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
            if (cldnn::format::dimension(input_node.get_output_layout().format) == 5)
                continue;

            // Disable for blocked formats
            if ((_lo.get_optimization_attributes().b_fs_yx_fsv16_network || input_node.get_output_layout().format == format::b_fs_yx_fsv16) &&
                _lo.is_format_optimized(node->as<deconvolution>(), format::b_fs_yx_fsv16)) {
                continue;
            }


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
            auto output_padding = deconv_prim->output_padding;

            // remove deconvolution node and its connections to weights and biases, rename it and move to the optimized
            // list
            tensor filter_size = {1, 1, 1, 1, 1};
            p.remove_connection(node->get_dependency(0), *node);
            for (auto& weights_id : weights_vec) {
                auto weights_iter = p.nodes_map.find(weights_id);
                if (weights_iter == p.nodes_map.end())  continue;

                auto weights_node_ptr = weights_iter->second;
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
                    auto bias_iter = p.nodes_map.find(bias_id);
                    if (bias_iter == p.nodes_map.end())  continue;

                    auto bias_id_node_ptr = bias_iter->second;
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
                                                               output_padding);
                p.get_or_create(conv_prim);
            } else {
                auto conv_prim = std::make_shared<convolution>(deconv_id,
                                                               input_id,
                                                               weights_vec,
                                                               stride,
                                                               input_offset,
                                                               tensor{1, 1, 1, 1},
                                                               output_padding);
                p.get_or_create(conv_prim);
            }

            auto conv_node_itr = p.nodes_map.find(deconv_id);
            if (conv_node_itr == p.nodes_map.end()) continue;

            auto conv_node_ptr = conv_node_itr->second;
            auto conv_node = &conv_node_ptr->as<convolution>();
            conv_node->set_transposed(true);

            // add connections input->convolution, weights->convolution and bias->convolution
            p.add_connection(input_node, *conv_node_ptr);

            for (auto& weights_id : weights_vec) {
                auto weights_node_itr = p.nodes_map.find(weights_id);
                if (weights_node_itr == p.nodes_map.end()) continue;

                auto weights_node_ptr = weights_node_itr->second;
                p.add_connection(*weights_node_ptr, *conv_node_ptr);
            }

            if (!bias_vec.empty()) {
                for (auto& bias_id : bias_vec) {
                    auto bias_id_node_itr = p.nodes_map.find(bias_id);
                    if (bias_id_node_itr == p.nodes_map.end()) continue;

                    auto bias_id_node_ptr = bias_id_node_itr->second;
                    p.add_connection(*bias_id_node_ptr, *conv_node_ptr);
                }
            }

            auto deconv_node_itr = p.nodes_map.find(rename_id);
            if (deconv_node_itr != p.nodes_map.end()) {
                auto deconv_node_ptr = deconv_node_itr->second;
                p.replace_all_usages(*deconv_node_ptr, *conv_node_ptr);
                p.optimized_out.push_back(rename_id);
                p.nodes_map.erase(rename_id);
            }

            p.mark_if_data_flow(*conv_node);
            conv_node->recalc_output_layout(true);

            update_processing_order = true;
        }
    }

    if (update_processing_order) {
        p.get_processing_order().calc_processing_order(p);
    }
}
