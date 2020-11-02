/*
// Copyright (c) 2016 Intel Corporation
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

#include "concatenation_inst.h"
#include "primitive_type_base.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>
#include <vector>
#include <memory>
#include <list>

namespace cldnn {
primitive_type_id concatenation::type_id() {
    static primitive_type_base<concatenation> instance;
    return &instance;
}

layout concatenation_inst::calc_output_layout(concatenation_node const& node) {
    auto desc = node.get_primitive();

    auto input_layout = node.input(0).get_output_layout();
    auto output_format = input_layout.format;
    auto result_sizes = input_layout.size.sizes();

    auto output_dt = desc->output_data_type ? *desc->output_data_type : input_layout.data_type;

    auto axis_index = node.get_primitive()->axis;

    // calculate sum of features from all inputs
    result_sizes[axis_index] = 0;
    for (size_t i = 0; i < desc->input.size(); ++i) {
        auto input_sizes = node.input(i).get_output_layout().size.sizes();
        if (node.input(i).get_output_layout().format == format::b_fs_yx_fsv16)
            output_format = format::b_fs_yx_fsv16;

        result_sizes[axis_index] += input_sizes[axis_index];
    }

    return layout {output_dt, output_format, (tensor) result_sizes};
}

std::string concatenation_inst::to_string(concatenation_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    std::stringstream ss_inputs;
    std::stringstream primitive_description;

    for (size_t i = 0; i < node.inputs_count(); ++i) {
        ss_inputs << node.input(i).id();
        ss_inputs << ", count: " << node.input(i).get_output_layout().count();
        i != (node.inputs_count() - 1) ? ss_inputs << ", " : ss_inputs << "";
    }

    json_composite concat_info;
    concat_info.add("concat axis", desc->axis);
    concat_info.add("inputs count", node.inputs_count());
    concat_info.add("inputs", ss_inputs.str());
    concat_info.dump(primitive_description);

    node_info->add("concat info", concat_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

concatenation_inst::typed_primitive_inst(network_impl& network, concatenation_node const& node)
    : parent(network, node) {
    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();

    tensor::value_type concat_count = 0;
    auto input_size = input_layout.size;
    auto output_size = output_layout.size;
    for (const auto& i : node.get_dependencies()) {
        auto input_i_layout = i->get_output_layout();
        auto input_mem_size = input_i_layout.size;
        for (int dim = concatenation::along_b; dim <= concatenation::along_w; ++dim) {
            if (dim == node.get_primitive()->axis) {
                concat_count += input_mem_size.raw[dim];
            } else {
                CLDNN_ERROR_NOT_EQUAL(node.id(),
                                      "Input size dim: " + std::to_string(dim),
                                      input_size.raw[dim],
                                      "input memory dim: " + std::to_string(dim),
                                      input_mem_size.raw[dim],
                                      "Every input must have the same size");
            }
        }
    }

    for (int dim = concatenation::along_b; dim <= concatenation::along_w; ++dim) {
        if (dim == node.get_primitive()->axis) {
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Concat count",
                                  concat_count,
                                  "output size dim:" + std::to_string(dim),
                                  output_size.raw[dim],
                                  "Output size in concatenated dimension mismatch sum of inputs!");
        } else {
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Input size dim: " + std::to_string(dim),
                                  input_size.raw[dim],
                                  "output size dim:" + std::to_string(dim),
                                  output_size.raw[dim],
                                  "Output size in non-concatenated dimension mistmatch input");
        }
    }

    if (node.can_be_optimized()) {
        build_deps();
        std::list<std::vector<std::shared_ptr<primitive_inst>>*> stack = {&_deps};
        while (!stack.empty()) {
            auto nodes_list = stack.front();
            stack.pop_front();

            for (auto processed_node : *nodes_list) {
                processed_node->_output = _output;
                if (processed_node->type() == concatenation::type_id() && processed_node->can_be_optimized()) {
                    if (!processed_node->_deps.empty())
                        stack.push_back(&processed_node->_deps);
                }
            }
        }
    }
}
}  // namespace cldnn
