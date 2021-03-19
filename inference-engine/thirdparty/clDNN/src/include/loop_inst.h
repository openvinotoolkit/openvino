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

// TODO(cldnn loop): clDNN/src/include/loop_inst.h loop_node loop_inst
namespace cldnn {
template<>
struct typed_program_node<loop> : public typed_program_node_base<loop> {
private:
    using parent = typed_program_node_base<loop>;
    using primitive_map_cref_t = std::map<primitive_id, std::reference_wrapper<const loop::primitive_mapping>>;
    using primitive_map_t = std::vector<loop::primitive_mapping>;
    topology_impl& body;

    primitive_map_cref_t input_primitive_map;
    primitive_map_cref_t output_primitive_map;
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
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog),
          body(*this->get_primitive()->body.get()),
          use_current_iteration(!this->get_primitive()->current_iteration_id.empty()),
          use_execution_condition(!this->get_primitive()->execution_condition_id.empty()) {
            input_primitive_map = this->get_primitive_mapping(this->get_primitive()->primitive_map, loop::INPUT);
            output_primitive_map = this->get_primitive_mapping(this->get_primitive()->primitive_map, loop::OUTPUT);
            output_need_concat = output_primitive_map.begin()->second.get().axis >= 0;
        }

    mutable int iteration_axis;

    program_impl::ptr get_body_program() const {
        return body_program;
    }

    bool is_output_working_as_backedge() const {
        return output_is_backedge;
    }

    bool is_current_iteration_used() const { return use_current_iteration; }
    bool is_execution_condition_used() const { return use_execution_condition; }
    bool need_output_concat() const { return output_need_concat; }

    static const loop::primitive_mapping* find_primitive_mapping(const primitive_id& external_id,
        const primitive_map_t& primitive_map,
        const loop::primitive_type type) {
        auto input_mapping = std::find_if(primitive_map.begin(),
                                       primitive_map.end(),
                                       [&](const loop::primitive_mapping& pm) {
                                           return (pm.type == type) && pm.external_id == external_id; });
        if (input_mapping == primitive_map.end())
            return nullptr;
        return &(*input_mapping);
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

    const primitive_map_cref_t& get_input_primitive_map() const { return input_primitive_map; }
    const primitive_map_cref_t& get_output_primitive_map() const { return output_primitive_map; }
    const primitive_map_t& get_primitive_map() const { return get_primitive()->primitive_map;}
    const std::vector<cldnn::loop::backedge_mapping>& get_back_edges() const { return get_primitive()->back_edges;}

    static primitive_map_cref_t get_primitive_mapping(const primitive_map_t& primitive_map, loop::primitive_type type) {
        primitive_map_cref_t ret;
        for (const auto& p : primitive_map) {
            if (p.type == type) {
                ret.emplace(p.external_id, std::cref(p));
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
            assert(input_primitive_map.count(id));
            const loop::primitive_mapping& input_rule = input_primitive_map.at(id).get();
            layout calculated_layout = calc_body_input_layout(input_rule);
            const primitive_id& internal_input_id = input_rule.internal_id;

            // add inputs for body network if not exist
            if (body.get_primitives().count(internal_input_id) == 0) {
                body.add(std::make_shared<input_layout>(internal_input_id, calculated_layout));
            } else {
                body.change_input_layout(internal_input_id, calculated_layout);
            }
        }

        // setup internal output
        std::set<primitive_id> output_names;
        const loop::primitive_mapping& output_mapping = output_primitive_map.begin()->second.get();
        output_is_backedge = false;
        const auto& back_edges = get_primitive()->back_edges;
        for (const auto& out : back_edges) {
            if (out.from == output_mapping.internal_id) {
                output_is_backedge = true;
                break;
            }
        }

        // add current_iteration_id in body network, execution_condition_id if exist
        process_single_int_input(get_current_iteration_id());
        process_single_int_input(get_execution_condition_id());

        // setup outputs for backedges
        for (auto& back_edge : back_edges) {
            const primitive_id& body_output_id = back_edge.from;

            for (const auto& prim : body.get_primitives()) {
                for (auto &prims_dep : prim.second->dependencies()) {
                    if (prims_dep.get() == back_edge.from && prim.first != body_output_id) {
                        prims_dep.get() = body_output_id;
                        break;
                    }
                }
            }
            const auto& input_mapping = std::find_if(input_primitive_map.begin(), input_primitive_map.end(),
                [&](const std::pair<cldnn::primitive_id, std::reference_wrapper<const cldnn::loop::primitive_mapping>>& pm) {
                    return pm.second.get().internal_id == back_edge.to;
            });
            assert(input_mapping != input_primitive_map.end());

            setup_internal_mutabledata_node(body_output_id, calc_body_input_layout(input_mapping->second), { back_edge.from });
            output_names.insert(body_output_id);
        }

        auto opts = get_program().get_options();
        std::vector<primitive_id> output_names_vec;
        for (auto name : output_names)
            output_names_vec.push_back(name);
        opts.set_option(build_option::outputs(output_names_vec));
        body_program = get_program().get_engine().build_program(body, opts, true);
    }

    const primitive_id& get_trip_count_id() const { return get_primitive()->trip_count_id; }
    const primitive_id& get_initial_execution_id() const { return get_primitive()->initial_execution_id; }
    const primitive_id& get_current_iteration_id() const { return get_primitive()->current_iteration_id; }
    const primitive_id& get_execution_condition_id() const { return get_primitive()->execution_condition_id; }
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
