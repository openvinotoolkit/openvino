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
#pragma once

#include <api/tensor_iterator.hpp>
#include <api/mutable_data.hpp>
#include <api/input_layout.hpp>
#include <api/memory.hpp>

#include "network_impl.h"
#include "primitive_inst.h"
#include <string>
#include <memory>
#include <vector>

namespace cldnn {
namespace details {}

template<>
struct typed_program_node<tensor_iterator> : public typed_program_node_base<tensor_iterator> {
private:
    using parent = typed_program_node_base<tensor_iterator>;
    topology_impl& body;

    mutable program_impl::ptr body_program;
    mutable std::map<primitive_id, memory_impl::ptr> backedge_mem_impls;
    mutable std::map<primitive_id, std::shared_ptr<mutable_data>> backedge_layers;
    mutable std::map<primitive_id, std::shared_ptr<memory>> backedge_mem;  
    
    mutable bool output_is_backedge;

    void setup_internal_mutabledata_node(primitive_id md_id, layout md_layout, std::vector<primitive_id> md_inputs_id = {}) const {
        if (body.get_primitives().count(md_id) == 0) {
            backedge_mem_impls[md_id] = get_program().get_engine().allocate_memory(md_layout, 0);
            backedge_mem[md_id] = std::make_shared<memory>(backedge_mem_impls[md_id].get());
            backedge_layers[md_id] = std::make_shared<mutable_data>(md_id, md_inputs_id, *backedge_mem[md_id]);       
            body.add(backedge_layers[md_id]);
        }
    }

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog),
          body(*this->get_primitive()->body.get()),
          ports_desc(get_primitive()->ports_desc){}

    mutable int iterations;
    mutable int iteration_axis;

    program_impl::ptr get_body_program() const {
        return body_program;
    };

    bool is_output_working_as_backedge() const {
        return output_is_backedge;
    }
    const std::string backedge_suffix = ":backedge";

    static const tensor_iterator::input_mapping* find_input_port_description(int prim_num, const tensor_iterator::port_map_collection& ports_desc) {
        auto input_desc = std::find_if(ports_desc.input_ports.begin(),
                                       ports_desc.input_ports.end(),
                                       [&](const tensor_iterator::input_mapping& im) { return im.from == prim_num; });
        if (input_desc == std::end(ports_desc.input_ports))
            return nullptr;
        return &(*input_desc);
     }

    layout calculate_layout_for_input(int input_num) const {
        auto* inputDesc = find_input_port_description(input_num, ports_desc);
        assert(inputDesc != nullptr);
        layout calculated_layout = get_dependency(input_num).get_output_layout();
        if (inputDesc->axis >= 0)
            calculated_layout.size.raw[iteration_axis] = 1;
        return calculated_layout;
    }

    void build_body_program() const {
        auto deps = get_dependencies();

        // setup internal inputs
        for (int i = 0; i < deps.size(); i++) {
            const tensor_iterator::input_mapping* input_rule = find_input_port_description(i, ports_desc);
            assert(desc != nullptr);
            layout calculated_layout = calculate_layout_for_input(i);
            if (body.get_primitives().count(input_rule->to) == 0)
                body.add(std::make_shared<input_layout>(input_rule->to, calculated_layout));
            else
                body.change_input_layout(input_rule->to, calculated_layout);
        }
        // setup internal output
        std::set<primitive_id> output_names;

        output_is_backedge = false;
        for (auto out : ports_desc.back_edges) {
            if (out.from == ports_desc.output_ports.at(0)) {
                output_is_backedge = true;
                break;
            }
        }

        if (!output_is_backedge)
            output_names.insert(ports_desc.output_ports.at(0));

        // setup outputs for backedges
        for (auto backedge : ports_desc.back_edges) {
            primitive_id from_with_suffix = backedge.from + backedge_suffix;

            for (auto prim : body.get_primitives()) {
                for (auto &prims_dep : prim.second->dependencies()) {
                    if (prims_dep.get() == backedge.from && prim.first != from_with_suffix) {
                        prims_dep.get() = from_with_suffix;
                        break;
                    }
                }
            }
            setup_internal_mutabledata_node(backedge.from + backedge_suffix, calculate_layout_for_input(backedge.to), { backedge.from });
            output_names.insert(backedge.from + backedge_suffix);
        }

        auto opts = get_program().get_options();
        std::vector<primitive_id> output_names_vec;
        for (auto name : output_names)
            output_names_vec.push_back(name);
        opts.set_option(build_option::outputs(output_names_vec));
        body_program = get_program().get_engine().build_program(body, opts, true);
    }

    std::vector<primitive_id> body_inputs() const {
        std::vector<primitive_id> ids;
        for (auto& dep_id : get_dependencies_ids()) {
            ids.push_back(id() + ":" + dep_id);
        }
        return ids;
    }
    const tensor_iterator::port_map_collection &ports_desc;
};

using tensor_iterator_node = typed_program_node<tensor_iterator>;

template <>
class typed_primitive_inst<tensor_iterator> : public typed_primitive_inst_base<tensor_iterator> {
    using parent = typed_primitive_inst_base<tensor_iterator>;

public:
    static layout calc_output_layout(tensor_iterator_node const& node);
    static std::string to_string(tensor_iterator_node const& node);
    typed_primitive_inst(network_impl& network, tensor_iterator_node const& node);
    network_impl::ptr get_body_network() const { return body_network; };
    std::vector<primitive_id> body_inputs() const { return node.body_inputs(); }
    const tensor_iterator::port_map_collection get_port_desc() const { return node.ports_desc; };
private:
    network_impl::ptr body_network;
};

using tensor_iterator_inst = typed_primitive_inst<tensor_iterator>;
}  // namespace cldnn
