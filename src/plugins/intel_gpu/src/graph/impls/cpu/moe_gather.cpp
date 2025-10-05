// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "moe_gather_inst.h"
#include "registry/implementation_map.hpp"

namespace cldnn {
namespace cpu {

struct moe_gather_impl : public typed_primitive_impl<moe_gather> {
    using parent = typed_primitive_impl<moe_gather>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::moe_gather_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_gather_impl>(*this);
    }

    moe_gather_impl() : parent("moe_gather_cpu_impl") {}

    explicit moe_gather_impl(const moe_gather_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<moe_gather>(), "[GPU] Incorrect program_node type");
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, moe_gather_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "moe_gather::execute_impl");
        auto& stream = instance.get_network().get_stream();

        if (instance.can_be_optimized()) {
            return stream.group_events(events);
        }

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();
        if (!pass_through_events) {
            stream.wait_for_events(events);
        }
        auto input_activations_mem_ptr = instance.dep_memory_ptr(0);
        auto experts_info_offsets_mem_ptr = instance.dep_memory_ptr(1);
        auto tokens_per_expert_mem_ptr = instance.dep_memory_ptr(2);
        auto tokens_len_per_expert_mem_ptr = instance.dep_memory_ptr(3);
        auto out_mem_ptr = instance.output_memory_ptr(0);
        auto num_used_experts = instance.get_input_layout(1).get_shape()[0];
        auto hidden_size = instance.get_input_layout(0).get_shape()[1];
        cldnn::mem_lock<ov::float16, mem_lock_type::read> input_data(input_activations_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::read> experts_info_offsets(experts_info_offsets_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::read> tokens_per_expert(tokens_per_expert_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::read> tokens_len_per_expert(tokens_len_per_expert_mem_ptr, stream);
        cldnn::mem_lock<ov::float16, mem_lock_type::read_write> output(out_mem_ptr, stream);

        for (size_t e = 0; e < num_used_experts; ++e) {
            size_t expert_info_offset = experts_info_offsets[e];
            size_t tokens_len = tokens_len_per_expert[e];
            for (size_t t = 0; t < tokens_len; ++t) {
                size_t token_id = tokens_per_expert[expert_info_offset + t];
                for (size_t h = 0; h < hidden_size; ++h) {
                    auto input_idx = token_id * hidden_size + h;
                    auto output_idx = (expert_info_offset + t) * hidden_size + h;
                    output[output_idx] = input_data[input_idx];
                }
            }
        }

        if (pass_through_events) {
            return stream.group_events(events);
        }

        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const moe_gather_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<moe_gather_impl>();
    }
};


namespace detail {

attach_moe_gather_impl::attach_moe_gather_impl() {
    auto formats = {
        format::bfyx,
    };

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i64,
        data_types::i8,
        data_types::u8,
    };

    implementation_map<moe_gather>::add(impl_types::cpu, shape_types::static_shape, moe_gather_impl::create, types, formats);
    implementation_map<moe_gather>::add(impl_types::cpu, shape_types::dynamic_shape, moe_gather_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::moe_gather_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gather)
