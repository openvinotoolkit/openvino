// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/condition.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
namespace details {}

template <>
struct typed_program_node<condition> : public typed_program_node_base<condition> {
private:
    using parent = typed_program_node_base<condition>;

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<condition> prim, program& prog)
        : parent(prim, prog),
          _branch_true(prim->branch_true),
          _branch_false(prim->branch_false) {}

    condition::branch get_branch_true() const { return _branch_true; }
    condition::branch get_branch_false() const { return _branch_false; }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        params->inner_progs = { _branch_true.inner_program, _branch_false.inner_program };
        params->io_output_maps = { _branch_true.output_map, _branch_false.output_map };
        return params;
    }

    void update_primitive_map(const primitive_id& prevID, const primitive_id& newID) {
        auto replace_external_id = [&](std::map<primitive_id, primitive_id>& input_map, const primitive_id& prevID, const primitive_id& newID) {
            auto iter = input_map.find(prevID);
            if (iter != input_map.end()) {
                primitive_id new_external_id = newID;
                primitive_id internal_id = iter->second;
                input_map.erase(iter);
                input_map.insert({new_external_id, internal_id});
                std::cout << "Replace id from " << prevID << " to " << newID << std::endl;
            }
        };

        std::cout << "-----------------------------------------------------------" << std::endl;
        std::cout << "Update primitive map ... " << std::endl;
        replace_external_id(_branch_true.input_map, prevID, newID);
        replace_external_id(_branch_false.input_map, prevID, newID);

        auto debug_input_map = [&](const std::map<primitive_id, primitive_id>& input_maps, std::string title) {
            std::cout << "Checking input map for [" << title << "]" << std::endl;
            for (const auto& ma : input_maps) {
                std::cout << "* external: " << ma.first << " == internal: " << ma.second << std::endl;
            }
        };
        debug_input_map(_branch_true.input_map, "branch_true");
        debug_input_map(_branch_false.input_map, "branch_false");
        std::cout << "-----------------------------------------------------------" << std::endl;
    }

private:
    // mutable condition::branch _branch_true;
    // mutable condition::branch _branch_false;
    condition::branch& _branch_true;
    condition::branch& _branch_false;
};

using condition_node = typed_program_node<condition>;

template <>
class typed_primitive_inst<condition> : public typed_primitive_inst_base<condition> {
    using parent = typed_primitive_inst_base<condition>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(condition_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(condition_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(condition_node const& node);
    typed_primitive_inst(network& network, condition_node const& node);

    memory::ptr compare_memory_ptr() const { return dep_memory_ptr(0); }
    network::ptr get_net_true() const { return _net_true; }
    network::ptr get_net_false() const { return _net_false; }
    network::ptr get_inner_networks(bool is_net_true);

private:
    network::ptr _net_true;
    network::ptr _net_false;
};

using condition_inst = typed_primitive_inst<condition>;
}  // namespace cldnn
