// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/moe_expert.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>
#include <vector>

namespace cldnn {
namespace details {}

struct expert_mask_scratch {
    std::vector<int8_t> pred_flag;
    // shape: [expert_num, batch_no]
    std::vector<std::vector<int>> batch;
    // shape: [expert_num, topk_no]
    std::vector<std::vector<int>> topk;
    size_t execed_count = 0;
};
struct expert_mask_mem_scratch {
    memory::ptr batch;
    memory::ptr topk;
    size_t max_size = 0;
};
static constexpr const char* expert_mask_scratch_key = "expert_mask_scratch";
static constexpr const char* expert_mask_mem_scratch_key = "expert_mask_scratch_mem";

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
    static void get_expert_mask_from_memory(memory::ptr mem, layout& layout, stream& stream, expert_mask_scratch& expert_mask);
    static bool get_pred_from_memory(memory::ptr mem, stream& stream, size_t expert_no);
    typed_primitive_inst(network& network, moe_expert_node const& node);

    memory::ptr pred_memory_ptr() const { return dep_memory_ptr(1); }
    const primitive_inst* pred_inst() const { return dependencies().at(1).first; }
    network::ptr get_net() const { return _net; }
    moe_expert::branch get_branch() const { return node->get_branch(); }
    const MOEExpert::Config& get_config() const {
        return node->get_primitive()->_config;
    }
    void copy_expert_mask_to_gpu(stream& stream,
                                 const expert_mask_scratch& expert_mask,
                                 size_t expert_no,
                                 expert_mask_mem_scratch& expert_mask_mem);

    void update_output_layout();
    void postprocess_output_memory(network::ptr executed_net, cldnn::moe_expert::branch& branch);

private:
    network::ptr _net;
};

using moe_expert_inst = typed_primitive_inst<moe_expert>;
}  // namespace cldnn
