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

struct expert_mask_tmp_scratch {
    memory::ptr x;
    memory::ptr up;
    memory::ptr gate;
    memory::ptr y;
    memory::ptr routing_weights;
    memory::ptr expert_info;

    memory::ptr topk_id;
    memory::ptr topk_weights;
    layout topk_id_layout;
    layout topk_weights_layout;
    int topk_size = 0;

    layout x_layout;
    size_t max_size = 0;
};

struct expert_mask_output_scratch {
    memory::ptr buf;
    layout buf_layout;
    size_t max_size = 0;
};

// one expert weights pointers
struct expert_info {
    void* weight[3];
    void* zp[3];
    void* scale[3];
    int routing_offset;
    int pad;
};

static constexpr const char* expert_mask_mem_scratch_key = "expert_mask_scratch_mem";
static constexpr const char* expert_mask_tmp_scratch_key = "expert_mask_scratch_tmp";
static constexpr const char* expert_mask_output_scratch_key = "expert_mask_scratch_output";

template <>
struct typed_program_node<moe_expert> : public typed_program_node_base<moe_expert> {
private:
    using parent = typed_program_node_base<moe_expert>;

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<moe_expert> prim, program& prog) : parent(prim, prog), _mlp_params(prim->_mlp_params) {}

    const std::vector<moe_expert::mlp_params>& get_mlp_params() const {
        return _mlp_params;
    }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);

        return params;
    }

private:
    std::vector<moe_expert::mlp_params>& _mlp_params;
};

using moe_expert_node = typed_program_node<moe_expert>;

template <>
class typed_primitive_inst<moe_expert> : public typed_primitive_inst_base<moe_expert> {
    using parent = typed_primitive_inst_base<moe_expert>;
    using parent::parent;
    using primitive_inst::update_output_memory;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(moe_expert_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(moe_expert_node const& /* node */, kernel_impl_params const& impl_param);
    static std::string to_string(moe_expert_node const& node);
    void get_expert_mask_from_memory(memory::ptr mem, layout& layout, stream& stream, expert_mask_scratch& expert_mask);
    static bool get_pred_from_memory(memory::ptr mem, stream& stream, size_t expert_no);
    typed_primitive_inst(network& network, moe_expert_node const& node);

    memory::ptr pred_memory_ptr() const { return dep_memory_ptr(1); }
    const primitive_inst* pred_inst() const { return dependencies().at(1).first; }
    const std::vector<moe_expert::mlp_params>& get_mlp_params() const { return node->get_mlp_params(); }
    const MOEExpert::Config& get_config() const {
        return node->get_primitive()->_config;
    }
    void copy_expert_mask_to_gpu(stream& stream,
                                 const expert_mask_scratch& expert_mask,
                                 size_t expert_no,
                                 expert_mask_mem_scratch& expert_mask_mem);

    void update_output_layout();
    void update_output_memory(bool need_reset);
    void get_tmp_memory(data_types type, int m, int hidden_size, int inter_size, int topk, expert_mask_tmp_scratch& scratch);

    memory::ptr alloc_buf(memory* curr_memory, layout& alloc_layout, allocation_type alloc_type = cldnn::allocation_type::usm_device);
    memory::ptr reinterpret_buf(const memory& curr_memory, const layout& new_layout);
};

using moe_expert_inst = typed_primitive_inst<moe_expert>;
}  // namespace cldnn
