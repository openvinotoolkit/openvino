/*
// Copyright (c) 2020 Intel Corporation
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
#pragma once
#include "primitive.hpp"
#include "topology.hpp"
#include <vector>
#include <map>

namespace cldnn {

struct tensor_iterator : public primitive_base<tensor_iterator> {
    CLDNN_DECLARE_PRIMITIVE(tensor_iterator)

    struct input_mapping {
        input_mapping(int from, primitive_id to, int axis = -1)
            : from(from), to(to), axis(axis) {}
        int from;
        primitive_id to;
        int axis;
    };

    struct backedge_mapping {
        backedge_mapping(primitive_id from, int to)
            : from(from), to(to) {}
        primitive_id from;
        int to;
    };

    struct port_map_collection {
        port_map_collection(std::vector<input_mapping> input_ports, std::vector<primitive_id> output_ports, std::vector<backedge_mapping> back_edges)
            : input_ports(input_ports),
                output_ports(output_ports),
                back_edges(back_edges) {};
        std::vector<input_mapping> input_ports;
        std::vector<primitive_id> output_ports;
        std::vector<backedge_mapping> back_edges;
        int find_input_port_with_selected_axis() const {
            for (int i = 0; i < input_ports.size(); i++) {
                if (input_ports[i].axis >= 0) {
                    return i;
                }
            }
            return -1;
        }
    } ports_desc;

    tensor_iterator(const primitive_id& id,
        const std::vector<primitive_id> inputs,
        const topology& body,
        std::vector<input_mapping> input_mapping,
        std::vector<primitive_id> outputs,
        std::vector<backedge_mapping> backedge_mapping = {},
        const padding& output_padding = padding())
            : primitive_base(id, inputs, output_padding),
               body(body),
        ports_desc(input_mapping, outputs, backedge_mapping) {
        /*output_data_type = data_types::f16;*/
    };

    topology body;

    
};
}  // namespace cldnn
