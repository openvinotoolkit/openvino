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

#include <src/include/error_handler.h>
#include "pass_manager.h"
#include "program_helpers.h"
#include "strided_slice_inst.h"
#include "reshape_inst.h"
#include "data_inst.h"
#include <vector>
#include <memory>

using namespace cldnn;

void strided_slice_optimize::run(program_impl& p) {
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = (*node_itr++);
        if (node->is_type<strided_slice>()) {
            auto& strided_slice_node = node->as<strided_slice>();
            auto& new_axis_mask = strided_slice_node.get_primitive()->new_axis_mask;

            if (std::find(new_axis_mask.begin(), new_axis_mask.end(), 1) == new_axis_mask.end())
                continue;

            auto& deps = node->get_dependencies();
            for (size_t i = deps.size(); i--;)
                if (deps[i]->is_type<data>())
                    node->remove_dependency(i);

            auto node_layout = strided_slice_node.get_output_layout();
            auto node_size = node_layout.size.sizes(format::bfyx);

            auto is_shift_possible = [&](const std::vector<int32_t>& dims) -> bool {
                if (dims.empty())
                    CLDNN_ERROR_MESSAGE(node->id(), "Error while adding new axis: node has incorrect dimensions");

                if (dims[dims.size() - 1] == 1)
                    return true;
                else
                    CLDNN_ERROR_MESSAGE(node->id(), "Not supported yet: too many axes for adding");
                return false;
            };

            std::vector<int32_t> output_dims_sizes = node_size;
            if (std::find(new_axis_mask.begin(), new_axis_mask.end(), 1) != new_axis_mask.end()) {
                for (size_t i = 0; i < new_axis_mask.size(); ++i) {
                    if (new_axis_mask[new_axis_mask.size() - i - 1] == 1) {
                        if (is_shift_possible(output_dims_sizes)) {
                            for (size_t j = output_dims_sizes.size() - 1; j > i; --j)
                                output_dims_sizes[j] = output_dims_sizes[j - 1];
                            output_dims_sizes[i] = 1;
                        }
                    }
                }
            }

            auto reshape_prim = std::make_shared<reshape>(
                "reshape_" + node->id(),
                node->get_dependency(0).get_primitive()->id,
                tensor(output_dims_sizes[0], output_dims_sizes[1], output_dims_sizes[3], output_dims_sizes[2]));

            auto& reshape_prim_node = p.get_or_create(reshape_prim);

            reshape_prim_node.set_output_layout(
                {node_layout.data_type, node_layout.format, reshape_prim->output_shape});

            p.add_intermediate(reshape_prim_node, *node, 0, true);
            p.extract_and_remove(*node);
        }
    }
}
