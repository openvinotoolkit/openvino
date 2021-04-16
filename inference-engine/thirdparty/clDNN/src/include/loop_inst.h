/*
// Copyright (c) 2021 Intel Corporation
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

#include "api/loop.hpp"
#include "api/mutable_data.hpp"
#include "api/input_layout.hpp"
#include "api/memory.hpp"

#include "network_impl.h"
#include "primitive_inst.h"
#include <string>
#include <memory>
#include <vector>

namespace cldnn {
template<>
struct typed_program_node<loop> : public typed_program_node_base<loop> {
private:
    using parent = typed_program_node_base<loop>;
    using primitive_map_t = std::vector<loop::primitive_mapping>;
    topology body_topology;
    topology_impl& body;

    primitive_map_t input_primitive_map;
    primitive_map_t output_primitive_map;
    std::vector<cldnn::loop::backedge_mapping> back_edges;
    bool use_current_iteration;
    bool use_execution_condition;
    bool output_need_concat;
    mutable program_impl::ptr body_program;
    mutable std::map<primitive_id, memory_impl::ptr> backedge_mem_impls;
    mutable std::map<primitive_id, std::shared_ptr<mutable_data>> backedge_layers;
    mutable std::map<primitive_id, std::shared_ptr<memory>> backedge_mem;

    mutable bool output_is_backedge;

    void setup_internal_mutabledata_node(primitive_id md_id, layout md_layout, std::vector<primitive_id> md_inputs_id = {}, uint32_t net_id = 0) const {
        if (body.get_primitives().count(md_id) == 0) {
            backedge_mem_impls[md_id] = get_program().get_engine().allocate_memory(md_layout, net_id);
            backedge_mem[md_id] = std::make_shared<memory>(backedge_mem_impls[md_id].get());
            backedge_layers[md_id] = std::make_shared<mutable_data>(md_id, md_inputs_id, *backedge_mem[md_id]);
            body.add(backedge_layers[md_id]);
        }
    }

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog) :
        parent(prim, prog),
        body_topology(this->get_primitive()->body),
        body(*body_topology.get()),
        back_edges(this->get_primitive()->back_edges),
        use_current_iteration(!this->get_primitive()->current_iteration_id.empty()),
        use_execution_condition(!this->get_primitive()->condition_id.empty()),
        max_iteration(this->get_primitive()->max_iteration < 0?
                this->get_primitive()->DEFAULT_MAX_ITERATION :
                this->get_primitive()->max_iteration) {
            input_primitive_map = this->get_primitive_mapping(this->get_primitive()->primitive_map, loop::INPUT);
            output_primitive_map = this->get_primitive_mapping(this->get_primitive()->primitive_map, loop::OUTPUT);
            output_need_concat = output_primitive_map.front().axis >= 0;
        }

    mutable int iteration_axis;
    int32_t max_iteration;

    int32_t get_max_iteration() const { return max_iteration; }
    // const std::vector<primitive_id>& get_loop_output_ids() const { return get_primitive()->outputs; }
    program_impl::ptr get_body_program() const { return body_program; }
    bool is_output_working_as_backedge() const { return output_is_backedge; }
    bool is_current_iteration_used() const { return use_current_iteration; }
    bool is_execution_condition_used() const { return use_execution_condition; }
    bool need_output_concat() const { return output_need_concat; }

    static std::vector<const loop::primitive_mapping*> find_primitive_mappings(
            const primitive_id& external_id,
            const primitive_map_t& primitive_map) {
        std::vector<const loop::primitive_mapping*> ret;
        for (auto it = primitive_map.begin(); it != primitive_map.end(); ++it) {
            if (it->external_id == external_id) {
                ret.push_back(&(*it));
            }
        }
        return ret;
    }

    static size_t convert_to_raw_axis(const int axis, const int ndim) {
        // convert between bfyx, bfzyx, bfzyxw and tensor.size.raw
        assert(axis < ndim);
        if (axis < 2) {
            return axis;
        }
        return (ndim-1)-(axis-2);
    }

    layout calc_body_input_layout(const loop::primitive_mapping& inputDesc) const {
        const auto& dependencies = get_dependencies();
        auto input = std::find_if(dependencies.begin(), dependencies.end(), [&inputDesc](const program_node* p){
            return p->id() == inputDesc.external_id;
        });
        assert(input != dependencies.end());
        layout calculated_layout = (*input)->get_output_layout();
        auto shape = calculated_layout.size.sizes(calculated_layout.format);

        if (inputDesc.axis >= 0) {
            iteration_axis = convert_to_raw_axis(inputDesc.axis, shape.size());
            calculated_layout.size.raw[iteration_axis] = 1; // cropped inputs shape
        }

        return calculated_layout;
    }

    const primitive_map_t& get_input_primitive_map() const { return input_primitive_map; }
    const primitive_map_t& get_output_primitive_map() const { return output_primitive_map; }

    void update_primitive_map(const primitive_id& prevID, const primitive_id& newID, bool external_id = true) {
        if (external_id) {
            for (auto& pm : input_primitive_map) {
                if (pm.external_id == prevID) {
                    pm.external_id = newID;
                    return;
                }
            }
            for (auto& pm : output_primitive_map) {
                if (pm.external_id == prevID) {
                    pm.external_id = newID;
                    return;
                }
            }
        } else {
            for (auto& pm : input_primitive_map) {
                if (pm.internal_id == prevID) {
                    pm.internal_id = newID;
                    return;
                }
            }
            for (auto& pm : output_primitive_map) {
                if (pm.internal_id == prevID) {
                    pm.internal_id = newID;
                    return;
                }
            }
            for (auto& back_edge : back_edges) {
                if (back_edge.from == prevID) {
                    back_edge.from = newID;
                    return;
                }
                if (back_edge.to == prevID) {
                    back_edge.to = newID;
                    return;
                }
            }
        }
    }

    const std::vector<cldnn::loop::backedge_mapping>& get_back_edges() const { return back_edges;}

    static primitive_map_t get_primitive_mapping(const primitive_map_t& primitive_map, loop::primitive_type type) {
        primitive_map_t ret;
        for (const auto& p : primitive_map) {
            if (p.type == type) {
                ret.push_back(p);
            }
        }
        return ret;
    }

    static bool is_integer(const data_types& data_type) {
        switch (data_type) {
            case data_types::i8:
            case data_types::i32:
            case data_types::i64:
                return true;
            default:
                return false;
        }
        return false;
    }

    void process_single_int_input(const primitive_id& id) const {
        const topology_map& body_topology_map = body.get_primitives();
        if (!id.empty()) {
            // add input_layout if not exist
            if (body_topology_map.count(id)) {
                layout body_input_layout(data_types::i32, format::bfyx, {1, 1, 1, 1});
                body.add(std::make_shared<input_layout>(id, body_input_layout));
            } else {
                const auto& body_input_prim = body.at(id);
                CLDNN_ERROR_BOOL(this->id(), "Error while building body program",
                    body_input_prim->type != input_layout::type_id(),
                    id + " is not cldnn::input_layout");
                const auto input_layout_prim = static_cast<const input_layout*>(body_input_prim.get());
                CLDNN_ERROR_BOOL(this->id(), "Error while building body program",
                    !static_cast<bool>(input_layout_prim->output_data_type),
                    "data_type of " + id + " is not specified");
                CLDNN_ERROR_BOOL(this->id(), "Error while building body program",
                    !is_integer(*input_layout_prim->output_data_type),
                    id + " is not integer type");
                CLDNN_ERROR_BOOL(this->id(), "Error while building body program",
                    input_layout_prim->layout.count() != 1,
                    id + " should have 1 element");
            }
        }
    }

    void build_body_program() const {
        const std::vector<cldnn::program_node *>& deps = get_dependencies();
        // setup internal inputs
        const primitive_id& trip_count_id = get_trip_count_id();
        const primitive_id& initial_execution = get_initial_execution_id();
        const primitive_id& num_iteration = get_num_iteration_id();
        for (const cldnn::program_node * dep : deps) {
            const primitive_id& id = dep->id();
            if (id == trip_count_id || id == initial_execution || id == num_iteration) {
                continue;
            }
            auto input_rules = find_primitive_mappings(id, input_primitive_map);
            assert(input_rules.size() > 0);
            for (const cldnn::loop::primitive_mapping * input_rule : input_rules) {
                layout calculated_layout = calc_body_input_layout(*input_rule);
                const primitive_id& internal_input_id = input_rule->internal_id;

                // add inputs for body network if not exist
                if (body.get_primitives().count(internal_input_id) == 0) {
                    body.add(std::make_shared<input_layout>(internal_input_id, calculated_layout));
                } else {
                    body.change_input_layout(internal_input_id, calculated_layout);
                }
            }
        }

        // setup internal output
        // TODO: handle multiple output_primitive_map
        std::set<primitive_id> output_names;
        output_names.insert(output_primitive_map.begin()->internal_id);
        const auto& back_edges = get_primitive()->back_edges;

        // add current_iteration_id in body network, condition_id if exist
        process_single_int_input(get_current_iteration_id());
        process_single_int_input(get_condition_id());

        // setup outputs for backedges
        for (auto& back_edge : back_edges) {
            // check whether the back_edge.to has its corresponding primitive_mapping
            const auto& input_mapping = std::find_if(input_primitive_map.begin(), input_primitive_map.end(),
                [&](const loop::primitive_mapping& pm) {
                    return pm.internal_id == back_edge.to;
                });
            if (input_mapping == input_primitive_map.end()) {
                std::string msg = "No primitive mapping for backedge (internal_id: " + back_edge.to + ')';
                CLDNN_ERROR_MESSAGE(this->id(), msg.c_str());
            }

            for (const auto& prim : body.get_primitives()) {
                if (prim.first != back_edge.from) {
                    continue;
                }
                const auto dependencies_ref = prim.second->dependencies();
                std::vector<primitive_id> dependencies(dependencies_ref.size());
                for (const auto& dep : dependencies_ref) {
                    dependencies.emplace_back(dep.get());
                }
                setup_internal_mutabledata_node(back_edge.from, calc_body_input_layout(*input_mapping), dependencies);
            }

            output_names.insert(back_edge.from);
        }

        auto opts = get_program().get_options();
        std::vector<primitive_id> output_names_vec(output_names.begin(), output_names.end());
        opts.set_option(build_option::outputs(output_names_vec));
        body_program = get_program().get_engine().build_program(body, opts, false);
    }

    const primitive_id& get_trip_count_id() const { return get_primitive()->trip_count_id; }
    const primitive_id& get_initial_execution_id() const { return get_primitive()->initial_execution_id; }
    const primitive_id& get_current_iteration_id() const { return get_primitive()->current_iteration_id; }
    const primitive_id& get_condition_id() const { return get_primitive()->condition_id; }
    const primitive_id& get_num_iteration_id() const { return get_primitive()->num_iteration_id; }
    const topology& get_body_topology() const { return get_primitive()->body; }
};

using loop_node = typed_program_node<loop>;

template <>
class typed_primitive_inst<loop> : public typed_primitive_inst_base<loop> {
    using parent = typed_primitive_inst_base<loop>;

public:
    static layout calc_output_layout(const loop_node& node);
    static std::string to_string(const loop_node& node);

public:
    typed_primitive_inst(network_impl& network, const loop_node& node);
    network_impl::ptr get_body_network() const { return body_network; }
private:
    network_impl::ptr body_network;
};

using loop_inst = typed_primitive_inst<loop>;
}  // namespace cldnn
