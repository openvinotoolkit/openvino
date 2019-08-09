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

#include <algorithm>

#include "pass_manager.h"
#include "program_node.h"
#include "mutable_data_inst.h"
#include "concatenation_inst.h"
#include "scale_inst.h"
#include "tensor_type.h"
#include <memory>

/*
This pass checks if data formats (layouts) of output/input in hidden layers match.
If not than required reorder is added to the network.
*/

/*
Add a reorder in between node and usr with reorder_layout as layout
*/
void add_required_reorders::add_reorder(program_impl& p, program_node* node, program_node* usr, layout reorder_layout) {
    auto new_reorder = std::make_shared<reorder>(node->id() + "_reorder_" + usr->id(), node->id(), reorder_layout);
    auto& new_reorder_node = p.get_or_create(new_reorder);

    // ToDo: add a method to program_impl class which adds an intermediate node given a node and its user
    auto it = std::find(usr->get_dependencies().begin(), usr->get_dependencies().end(), node);
    if (it == usr->get_dependencies().end()) {
        throw error("Inconcistency in topology description: user of a node is not present among its dependecies.",
                    CLDNN_ERROR);
    }
    auto idx = it - usr->get_dependencies().begin();
    if (idx < 0 || (size_t)idx >= usr->get_dependencies().size()) {
        throw error("Internal Error: container index out of range exception.", CLDNN_ERROR);
    }
    p.add_intermediate(new_reorder_node, *usr, idx);
}

void add_required_reorders::run(program_impl& p) {
    auto usr_itr = p.get_processing_order().begin();
    while (usr_itr != p.get_processing_order().end()) {
        auto& usr = *usr_itr++;
        if (usr->get_dependencies().size() == 0)
            continue;  // only nodes with dependencies
        if (usr->is_type<internal_primitive>() || usr->is_type<data>())
            continue;
        if (usr->type()->does_an_implementation_exist(p.get_engine(), *usr))
            continue;

        /*
            First check if there are non data flow dependencies for the primitive
            if so then choose the same output format as the data
        */
        bool correct_layout_selected = false;
        for (auto& node : usr->get_dependencies()) {
            if (!node->is_in_data_flow()) {
                /*
                    ToDo: Here we should handle also the situation where primitive usr has data inputs in different
                   formats
                */
                layout current_layout(usr->get_output_layout().data_type,
                                      node->get_output_layout().format,
                                      usr->get_output_layout().size);
                usr->set_output_layout(current_layout);
                if (usr->type()->does_possible_implementation_exist(p.get_engine(), *usr)) {
                    correct_layout_selected = true;
                    break;
                } else {
                    throw error("Internal Error: no layout format available for " + usr->id() + " comaptible with " +
                                    node->id(),
                                CLDNN_ERROR);
                }
            }
        }

        if (!correct_layout_selected) {
            // This list of preffered layouts has been selected arbitrary due to developers' experience
            cldnn::format preffered_layout_formats[]{
                // TODO: [block_formats] - verify if it is really needed
                cldnn::format::bfyx_f16,
                cldnn::format::bfyx,
                cldnn::format::yxfb,
                cldnn::format::byxf,
            };

            for (auto new_layout_format : preffered_layout_formats) {
                layout current_layout(usr->get_output_layout().data_type,
                                      new_layout_format,
                                      usr->get_output_layout().size);
                usr->set_output_layout(current_layout);
                if (usr->type()->does_possible_implementation_exist(p.get_engine(), *usr)) {
                    correct_layout_selected = true;
                    break;
                }
            }

            if (!correct_layout_selected) {
                throw error("Internal Error: no implementation for " + usr->id() +
                                " kernel which satisfies output format dependecies.",
                            CLDNN_ERROR);
            }
        }

        // layout is selected now add required reorders
        auto dep_itr = usr->get_dependencies().begin();
        while (dep_itr != usr->get_dependencies().end()) {
            auto node = *dep_itr++;
            // do not add a reorder if usr or node are reorders or does not belong to data_flow
            if (!usr->is_type<reorder>() && node->is_in_data_flow()) {
                if ((usr->get_output_layout() != node->get_output_layout())) {
                    add_reorder(p, node, usr, usr->get_output_layout());
                }
            }
        }
    }
}
