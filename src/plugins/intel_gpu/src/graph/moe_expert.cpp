// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_expert_inst.h"
#include "openvino/core/except.hpp"
#include "program_node.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "openvino/core/parallel.hpp"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(moe_expert)

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout moe_expert_inst::calc_output_layout(moe_expert_node const& /* node */, kernel_impl_params const& impl_param) {
    return impl_param.input_layouts[0];
}

bool moe_expert_inst::get_pred_from_memory(memory::ptr mem, stream& stream, size_t expert_no) {
    const auto& shape = mem->get_layout().get_shape();
    auto offset = expert_no * shape[1] * shape[2];
    mem_lock<int32_t, mem_lock_type::read> lock_data{mem, stream};
    auto p = lock_data.data() + offset;
    for (size_t i = 0; i < shape[1] * shape[2]; i++)
        if (p[i])
            return true;
    return false;
}

void moe_expert_inst::get_expert_mask_from_memory(memory::ptr mem, layout& layout, stream& stream, expert_mask_scratch& expert_mask) {
    // shape: [batch, topk]
    const auto& shape = layout.get_shape();
    const auto& config = get_config();

    int max_expert_num = static_cast<int>(config.expert_num),
        max_topk = static_cast<int>(config.topk),
        max_tokens = static_cast<int>(shape[0]);

    expert_mask.pred_flag.resize(max_expert_num, 0);
    expert_mask.batch.resize(max_expert_num, {});
    expert_mask.topk.resize(max_expert_num, {});

    OPENVINO_ASSERT(!layout.data_padding, "get_expert_mask_from_memory not support padding");

    std::vector<int32_t> buf(max_topk * max_tokens);
    mem->copy_to(stream, buf.data(), 0, 0, buf.size() * sizeof(int32_t), true);

    for (int b = 0; b < max_tokens; b++) {
        auto* tok_p = &buf[b * max_topk];
        for (int t = 0; t < max_topk; t++) {
            auto expert_no = tok_p[t];
            OPENVINO_ASSERT(expert_no < max_expert_num);
            expert_mask.batch[expert_no].push_back(b);
            expert_mask.topk[expert_no].push_back(t + b * max_topk);
            expert_mask.pred_flag[expert_no] = 1;
        }
    }
}

void moe_expert_inst::copy_expert_mask_to_gpu(stream& stream,
                                              const expert_mask_scratch& expert_mask,
                                              size_t expert_no,
                                              expert_mask_mem_scratch& expert_mask_mem) {
    layout new_layout(ov::PartialShape{static_cast<int>(expert_mask.batch[expert_no].size())}, ov::element::i32, cldnn::format::bfyx);
    auto new_size = expert_mask.batch[expert_no].size() * sizeof(int);

    if (new_size > expert_mask_mem.max_size) {
        auto cur_size = static_cast<int>(expert_mask.batch[expert_no].size());
        int will_allocate = std::max((cur_size + 255) / 256 * 256, 256);
        layout alloc_layout(ov::PartialShape{will_allocate}, ov::element::i32, cldnn::format::bfyx);
        auto alloc_type = _network.get_engine().get_lockable_preferred_memory_allocation_type();
        GPU_DEBUG_LOG << "=> allocate expert_mask to " << alloc_type << std::endl;
        auto& pool = _network.get_memory_pool();
        auto net_id = _network.get_id();

        auto alloc_buf = [&](memory* curr_memory) {
            if (_node->get_program().get_config().get_enable_memory_pool()) {
                if (curr_memory != nullptr)
                    pool.release_memory(curr_memory, _node->get_unique_id(), _node->id(), net_id);
                return pool.get_memory(alloc_layout,
                                       _node->id(),
                                       _node->get_unique_id(),
                                       net_id,
                                       _runtime_memory_dependencies,
                                       alloc_type,
                                       false,
                                       true,
                                       _node->is_dynamic());
            }
            return pool.get_memory(alloc_layout, alloc_type, true);
        };

        expert_mask_mem.batch = alloc_buf(expert_mask_mem.batch.get());
        expert_mask_mem.topk = alloc_buf(expert_mask_mem.topk.get());
        expert_mask_mem.max_size = expert_mask_mem.batch->size();
    }
    expert_mask_mem.batch = _network.get_engine().reinterpret_buffer(*expert_mask_mem.batch, new_layout);
    expert_mask_mem.topk = _network.get_engine().reinterpret_buffer(*expert_mask_mem.topk, new_layout);

    {
        mem_lock<int32_t, mem_lock_type::write> lock_data{expert_mask_mem.batch, stream};
        memcpy(lock_data.data(), expert_mask.batch[expert_no].data(), new_size);
    }
    {
        mem_lock<int32_t, mem_lock_type::write> lock_data{expert_mask_mem.topk, stream};
        memcpy(lock_data.data(), expert_mask.topk[expert_no].data(), new_size);
    }
}

void moe_expert_inst::get_tmp_memory(data_types type, int m, int hidden_size, int inter_size, int topk, expert_mask_tmp_scratch& scratch) {
    layout x_layout(ov::PartialShape{m, hidden_size}, type, cldnn::format::bfyx);

    if (x_layout == scratch.x_layout)
        return;

    auto new_size = x_layout.bytes_count();
    layout gate_layout(ov::PartialShape{m, inter_size}, type, cldnn::format::bfyx);
    layout routing_layout(ov::PartialShape{m * topk}, type, cldnn::format::bfyx);
    if (new_size > scratch.max_size) {
        auto& pool = _network.get_memory_pool();
        auto net_id = _network.get_id();

        auto alloc_buf = [&](memory* curr_memory, layout& alloc_layout, allocation_type alloc_type = cldnn::allocation_type::usm_device) {
            if (_node->get_program().get_config().get_enable_memory_pool()) {
                if (curr_memory != nullptr)
                    pool.release_memory(curr_memory, _node->get_unique_id(), _node->id(), net_id);
                return pool.get_memory(alloc_layout,
                                       _node->id(),
                                       _node->get_unique_id(),
                                       net_id,
                                       _runtime_memory_dependencies,
                                       alloc_type,
                                       false,
                                       true,
                                       _node->is_dynamic());
            }
            return pool.get_memory(alloc_layout, alloc_type, true);
        };
        if (!scratch.expert_info) {
            auto expected_alloc_type = _network.get_engine().get_lockable_preferred_memory_allocation_type();
            GPU_DEBUG_LOG << "=> allocate expert_mask to " << expected_alloc_type << std::endl;
            layout expert_layout(ov::PartialShape{topk * static_cast<int>(sizeof(expert_info))}, data_types::u8, cldnn::format::bfyx);
            scratch.expert_info = alloc_buf(nullptr, expert_layout, expected_alloc_type);

            const auto& config = get_config();
            layout ptr_layout(ov::PartialShape{config.expert_num}, data_types::u64, cldnn::format::byfx);
            scratch.gate_addrs = alloc_buf(nullptr, ptr_layout);
            scratch.gate_scales_addrs = alloc_buf(nullptr, ptr_layout);
            scratch.gate_zp_addrs = alloc_buf(nullptr, ptr_layout);
            scratch.up_addrs = alloc_buf(nullptr, ptr_layout);
            scratch.up_scales_addrs = alloc_buf(nullptr, ptr_layout);
            scratch.up_zp_addrs = alloc_buf(nullptr, ptr_layout);
            scratch.down_addrs = alloc_buf(nullptr, ptr_layout);
            scratch.down_scales_addrs = alloc_buf(nullptr, ptr_layout);
            scratch.down_zp_addrs = alloc_buf(nullptr, ptr_layout);
            std::array<std::vector<uint64_t>, 9> buf;
            for (size_t i = 0; i < get_mlp_params().size(); i++) {
                const auto& param = get_mlp_params()[i];
                buf[0].push_back(reinterpret_cast<uint64_t>(param.param[0].weight->buffer_ptr()));
                buf[1].push_back(reinterpret_cast<uint64_t>(param.param[0].scale_ba->buffer_ptr()));
                buf[2].push_back(reinterpret_cast<uint64_t>(param.param[0].zp_ba->buffer_ptr()));
                buf[3].push_back(reinterpret_cast<uint64_t>(param.param[1].weight->buffer_ptr()));
                buf[4].push_back(reinterpret_cast<uint64_t>(param.param[1].scale_ba->buffer_ptr()));
                buf[5].push_back(reinterpret_cast<uint64_t>(param.param[1].zp_ba->buffer_ptr()));
                buf[6].push_back(reinterpret_cast<uint64_t>(param.param[2].weight->buffer_ptr()));
                buf[7].push_back(reinterpret_cast<uint64_t>(param.param[2].scale_ba->buffer_ptr()));
                buf[8].push_back(reinterpret_cast<uint64_t>(param.param[2].zp_ba->buffer_ptr()));
            }
            scratch.gate_addrs->copy_from(_network.get_stream(), buf[0].data(), 0, 0, ptr_layout.bytes_count(), true);
            scratch.gate_scales_addrs->copy_from(_network.get_stream(), buf[1].data(), 0, 0, ptr_layout.bytes_count(), true);
            scratch.gate_zp_addrs->copy_from(_network.get_stream(), buf[2].data(), 0, 0, ptr_layout.bytes_count(), true);
            scratch.up_addrs->copy_from(_network.get_stream(), buf[3].data(), 0, 0, ptr_layout.bytes_count(), true);
            scratch.up_scales_addrs->copy_from(_network.get_stream(), buf[4].data(), 0, 0, ptr_layout.bytes_count(), true);
            scratch.up_zp_addrs->copy_from(_network.get_stream(), buf[5].data(), 0, 0, ptr_layout.bytes_count(), true);
            scratch.down_addrs->copy_from(_network.get_stream(), buf[6].data(), 0, 0, ptr_layout.bytes_count(), true);
            scratch.down_scales_addrs->copy_from(_network.get_stream(), buf[7].data(), 0, 0, ptr_layout.bytes_count(), true);
            scratch.down_zp_addrs->copy_from(_network.get_stream(), buf[8].data(), 0, 0, ptr_layout.bytes_count(), true);
        }

        scratch.x = alloc_buf(scratch.x.get(), x_layout);
        scratch.y = alloc_buf(scratch.y.get(), x_layout);
        scratch.up = alloc_buf(scratch.up.get(), gate_layout);
        scratch.gate = alloc_buf(scratch.gate.get(), gate_layout);
        scratch.routing_weights = alloc_buf(scratch.routing_weights.get(), routing_layout);
        scratch.max_size = scratch.x->size();
    }
    scratch.x = _network.get_engine().reinterpret_buffer(*scratch.x, x_layout);
    scratch.y = _network.get_engine().reinterpret_buffer(*scratch.y, x_layout);
    scratch.up = _network.get_engine().reinterpret_buffer(*scratch.up, gate_layout);
    scratch.gate = _network.get_engine().reinterpret_buffer(*scratch.gate, gate_layout);
    scratch.routing_weights = _network.get_engine().reinterpret_buffer(*scratch.routing_weights, routing_layout);
    scratch.x_layout = x_layout;
}

template<typename ShapeType>
std::vector<layout> moe_expert_inst::calc_output_layouts(moe_expert_node const& /* node */, kernel_impl_params const& impl_param) {
    return {impl_param.input_layouts[0]};
}

template std::vector<layout> moe_expert_inst::calc_output_layouts<ov::PartialShape>(moe_expert_node const& node, const kernel_impl_params& impl_param);

std::string moe_expert_inst::to_string(moe_expert_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite moe_expert_info;

    node_info->add("moe_expert info", moe_expert_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
moe_expert primitive is reusing memory with the input.
*/
moe_expert_inst::typed_primitive_inst(network& network, moe_expert_node const& node)
    : parent(network, node) {
}

void moe_expert_inst::update_output_layout() {
    for (size_t i = 0; i < _deps.size(); i++) {
        auto idx = _deps[i].second;
        auto new_shape = _deps[i].first->_impl_params->get_output_layout(idx);
        if (_impl_params->get_input_layout(i) != new_shape) {
            GPU_DEBUG_TRACE_DETAIL << id() << ": update shape dep [" << i << "] : " << _deps[i].first->id()
                                   << " was: " << _impl_params->get_input_layout(i).to_short_string()
                                   << " now: " << new_shape.to_short_string() << std::endl;
            _impl_params->input_layouts[i] = new_shape;
        }
    }
    auto memory_deps = _node->get_const_memory_deps();
    for (auto& i : _node->get_shape_infer_dependencies()) {
        if (memory_deps.count(i) > 0 || i >= _node->get_dependencies().size()) {
            continue;
        }
        auto dep_id = _node->get_dependency(i).id();

        auto dep_mem = _network.get_output_memory(dep_id);
        memory_deps.insert({i, dep_mem});
    }
    _impl_params->memory_deps = memory_deps;

    auto new_layouts = _node->type()->calc_output_layouts(*_node, *_impl_params);
    if (new_layouts.empty()) {
        auto new_layout = _node->type()->calc_output_layout(*_node, *_impl_params);
        new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(0), new_layout.data_padding);
        _impl_params->output_layouts[0] = new_layout;
    } else {
        for (size_t i = 0; i != new_layouts.size(); ++i) {
            auto new_layout = new_layouts[i];
            new_layout.data_padding = padding::max(_node->get_primitive()->get_output_padding(i), new_layout.data_padding);
            _impl_params->output_layouts[i] = new_layout;
        }
    }
}

void moe_expert_inst::update_output_memory(bool need_reset) {
    _outputs.resize(outputs_memory_count());

    _outputs[0] = input_memory_ptr(0);
    if (need_reset) {
        add_dep_event(_outputs[0]->fill(_network.get_stream(), false));
    }
}
}  // namespace cldnn
