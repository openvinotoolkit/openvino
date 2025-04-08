// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_expert_inst.h"
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
    const auto& shape = layout.get_shape();
    int max_expert_num = static_cast<int>(shape[0]), max_topk = static_cast<int>(shape[1]), max_tokens = static_cast<int>(shape[2]);
    expert_mask.pred_flag.clear();
    expert_mask.batch.clear();
    expert_mask.topk.clear();
    expert_mask.pred_flag.resize(max_expert_num, 0);
    expert_mask.batch.resize(max_expert_num, {});
    expert_mask.topk.resize(max_expert_num, {});
    auto num_per_expert = max_topk * max_tokens;
    auto fill = [&](int* p, int expert_no) {
        cldnn::tensor size = layout.get_tensor();
        // topk
        for (int f = 0; f < size.feature[0]; f++) {
            // bfxyzw
            auto offset = layout.get_linear_offset(cldnn::tensor(expert_no, f, 0, 0, 0, 0));
            // tokens
            for (int y = 0; y < size.spatial[1]; y++) {
                OPENVINO_ASSERT(static_cast<int>(offset) + y == expert_no * num_per_expert + f * max_tokens + y);
                if (p[offset + y]) {
                    expert_mask.pred_flag[expert_no] = 1;
                    expert_mask.batch[expert_no].push_back(y);
                    // routing weights is 1d, its shape: [topk * batch, 1] corresponding 2d shape: [batch, topk]. The following will get its offset in 1d:
                    //   batch * max_topk + topk
                    expert_mask.topk[expert_no].push_back(f + y * max_topk);
                }
            }
        }
    };
    //auto fill = [&](int* p, int expert_no) {
    //    auto offset = expert_no * num_per_expert;
    //    for (int t = 0; t < max_topk; t++) {
    //        for (int b = 0; b < max_tokens; b++) {
    //            if (p[offset + t * max_tokens + b]) {
    //                expert_mask.pred_flag[expert_no] = 1;
    //                expert_mask.batch[expert_no].push_back(b);
    //                // routing weights is 1d, its shape: [topk * batch, 1] corresponding 2d shape: [batch, topk]. The following will get its offset in 1d:
    //                //   batch * max_topk + topk
    //                expert_mask.topk[expert_no].push_back(t + b * max_topk);
    //            }
    //        }
    //    }
    //};
    mem_lock<int32_t, mem_lock_type::read> lock_data{mem, stream};
    auto p = lock_data.data();
    if (max_tokens < 5) {
        for (int expert_no = 0; expert_no < max_expert_num; expert_no++) {
            fill(p, expert_no);
        }
    } else {
        ov::parallel_for(max_expert_num, [&](int expert_no) {
            fill(p, expert_no);
        });
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
                                       false,
                                       _node->is_dynamic());
            }
            return pool.get_memory(alloc_layout, alloc_type, false);
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
    : parent(network, node),
      _net(network::allocate_network(network.get_stream_ptr(), node.get_branch().inner_program)) {
    this->set_inner_networks({_net});
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

void moe_expert_inst::postprocess_output_memory(network::ptr executed_net, cldnn::moe_expert::branch& branch) {
    _outputs.resize(outputs_memory_count());
    auto out_mem_idx = 0;
    auto inner_out_id = executed_net->get_output_ids()[0];

    auto mem_ptr = executed_net->get_output_memory(inner_out_id);
    if (mem_ptr) {
        auto layout = _impl_params->get_output_layout(out_mem_idx);
        GPU_DEBUG_LOG << "Reshape output from " << mem_ptr->get_layout().to_short_string()
                    << " to " << layout.to_short_string() << std::endl;
        // Preallocation logic may allocate more memory than actually produced on current iteration, so we need to adjust output buffers layout
        mem_ptr = get_network().get_engine().reinterpret_buffer(*mem_ptr, layout);
    }

    _outputs[out_mem_idx] = mem_ptr;
    if (mem_ptr)
        GPU_DEBUG_LOG << "Inner net - Outputs[" << out_mem_idx << "]" << mem_ptr->get_layout().to_short_string() << std::endl;
}
}  // namespace cldnn
