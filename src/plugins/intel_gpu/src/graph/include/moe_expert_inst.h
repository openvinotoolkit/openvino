// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/moe_expert.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
namespace details {}

template <>
struct typed_program_node<moe_expert> : public typed_program_node_base<moe_expert> {
private:
    using parent = typed_program_node_base<moe_expert>;

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<moe_expert> prim, program& prog)
        : parent(prim, prog),
          _branch(prim->_branch) {}

    moe_expert::branch get_branch() const { return _branch; }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        params->inner_progs = { _branch.inner_program };
        params->io_output_maps = { _branch.output_map };
        return params;
    }

private:
    moe_expert::branch& _branch;
};

using moe_expert_node = typed_program_node<moe_expert>;

template <>
class typed_primitive_inst<moe_expert> : public typed_primitive_inst_base<moe_expert> {
    using parent = typed_primitive_inst_base<moe_expert>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(moe_expert_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(moe_expert_node const& /* node */, kernel_impl_params const& impl_param);
    static std::string to_string(moe_expert_node const& node);
    static bool get_pred_from_memory(memory::ptr mem, stream& stream, size_t expert_no);
    typed_primitive_inst(network& network, moe_expert_node const& node);

    memory::ptr pred_memory_ptr() const { return dep_memory_ptr(1); }
    primitive_inst* pred_inst() { return _exec_deps[1]; }
    network::ptr get_net() const { return _net; }
    moe_expert::branch get_branch() const { return node->get_branch(); }
    const MOEExpert::Config& get_config() const {
        return node->get_primitive()->_config;
    }

    void update_output_layout();
    void postprocess_output_memory(network::ptr executed_net, cldnn::moe_expert::branch& branch);

private:
    network::ptr _net;
};

using moe_expert_inst = typed_primitive_inst<moe_expert>;
}  // namespace cldnn
