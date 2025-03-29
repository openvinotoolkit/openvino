// Copyright (C) 2018-2025 Intel Corporation
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
            }
        };

        replace_external_id(_branch_true.input_map, prevID, newID);
        replace_external_id(_branch_false.input_map, prevID, newID);
    }

private:
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
    static std::vector<layout> calc_output_layouts(condition_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(condition_node const& /* node */, kernel_impl_params const& impl_param);
    static std::string to_string(condition_node const& node);
    static bool get_pred_from_memory(memory::ptr mem, stream& stream);
    typed_primitive_inst(network& network, condition_node const& node);

    memory::ptr pred_memory_ptr() const { return dep_memory_ptr(0); }
    network::ptr get_net_true() const { return _net_true; }
    network::ptr get_net_false() const { return _net_false; }
    condition::branch get_branch_true() const { return node->get_branch_true(); }
    condition::branch get_branch_false() const { return node->get_branch_false(); }

    void update_output_layout();
    void postprocess_output_memory(network::ptr executed_net, cldnn::condition::branch& branch);

    static layout adjust_scalar_to_1d_layout(layout& target, layout& other);

private:
    network::ptr _net_true;
    network::ptr _net_false;
};

using condition_inst = typed_primitive_inst<condition>;
}  // namespace cldnn
