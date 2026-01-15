// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/cpu/cpu_impl_helpers.hpp"
#include "register.hpp"
#include "moe_mask_gen_inst.h"
#include "registry/implementation_map.hpp"

namespace cldnn {
namespace cpu {

struct moe_mask_gen_impl : public typed_primitive_impl<moe_mask_gen> {
    using parent = typed_primitive_impl<moe_mask_gen>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::moe_mask_gen_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_mask_gen_impl>(*this);
    }

    moe_mask_gen_impl() : parent("moe_mask_gen_cpu_impl") {}

    explicit moe_mask_gen_impl(const moe_mask_gen_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<moe_mask_gen>(), "[GPU] Incorrect program_node type");
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, moe_mask_gen_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "moe_mask_gen::execute_impl");
        auto& stream = instance.get_network().get_stream();

        const bool pass_through_events = (stream.get_queue_type() == QueueTypes::out_of_order) && instance.all_dependencies_cpu_impl();

        if (!pass_through_events) {
            stream.wait_for_events(events);
        }

        auto num_tokens            = instance.get_input_layout(0).get_shape()[0];
        auto num_experts_per_token = instance.get_node().as<moe_mask_gen>().get_primitive()->num_experts_per_token;
        auto num_total_experts     = instance.get_node().as<moe_mask_gen>().get_primitive()->num_total_experts;

        auto topk_idx_mem_ptr                 = instance.dep_memory_ptr(0);
        auto tokens_per_expert_mem_ptr        = instance.output_memory_ptr(0);
        auto experts_info_start_idx_mem_ptr   = instance.output_memory_ptr(1);
        auto experts_id_mem_ptr               = instance.output_memory_ptr(2);
        auto tokens_lens_per_expert_mem_ptr   = instance.output_memory_ptr(3);
        auto num_actual_used_experts_mem_ptr  = instance.output_memory_ptr(4);

        cldnn::mem_lock<int32_t, mem_lock_type::read> topk_idx_lock(topk_idx_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::write> num_actual_used_experts_lock(num_actual_used_experts_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::write> tokens_per_expert_lock(tokens_per_expert_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::write> experts_info_start_idx_lock(experts_info_start_idx_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::write> experts_id_lock(experts_id_mem_ptr, stream);
        cldnn::mem_lock<int32_t, mem_lock_type::write> tokens_lens_per_expert_lock(tokens_lens_per_expert_mem_ptr, stream);

        // make mask for gather
        std::vector<std::vector<int32_t>> tokens_per_expert(num_total_experts, std::vector<int32_t>());
        for (size_t token = 0; token < num_tokens; ++token) {
            for (int j = 0; j < num_experts_per_token; ++j) {
                const auto expert_id = topk_idx_lock[token * num_experts_per_token + j];
                tokens_per_expert[expert_id].push_back(static_cast<int32_t>(token));
            }
        }

        int tokens_per_expert_iter = 0;
        int experts_id_iter = 0;
        int num_actually_used_experts = 0;
        for (int expert = 0; expert < num_total_experts; expert++) {
            if (!tokens_per_expert[expert].empty()) {
                experts_info_start_idx_lock[experts_id_iter] = tokens_per_expert_iter;
                experts_id_lock[experts_id_iter] = expert;
                tokens_lens_per_expert_lock[experts_id_iter++] = static_cast<int32_t>(tokens_per_expert[expert].size());
                num_actually_used_experts++;
                for (auto t : tokens_per_expert[expert]) {
                    tokens_per_expert_lock[tokens_per_expert_iter++] = t;
                }
            }
        }
        num_actual_used_experts_lock[0] = num_actually_used_experts;

        if (pass_through_events) {
            return stream.group_events(events);
        }
        return make_output_event(stream, instance.is_output());
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const moe_mask_gen_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<moe_mask_gen_impl>();
    }
};

struct moe_mask_gen_reshape_impl : public typed_primitive_impl<moe_mask_gen_reshape> {
    using parent = typed_primitive_impl<moe_mask_gen_reshape>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::cpu::moe_mask_gen_reshape_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_mask_gen_reshape_impl>(*this);
    }

    moe_mask_gen_reshape_impl() : parent("moe_mask_gen_reshape_cpu_impl") {}

    explicit moe_mask_gen_reshape_impl(const moe_mask_gen_reshape_node& outer) {
        set_node_params(outer);
    }

    void set_node_params(const program_node& arg) override {
        OPENVINO_ASSERT(arg.is_type<moe_mask_gen_reshape>(), "[GPU] Incorrect program_node type");
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, moe_mask_gen_reshape_inst& instance) override {
        OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "moe_mask_gen_reshape::execute_impl");
        auto& stream = instance.get_network().get_stream();

        return stream.group_events(events);
    }

    void init_kernels(const kernels_cache& , const kernel_impl_params&) override {}

    void update(primitive_inst& inst, const kernel_impl_params& impl_param) override {}

public:
    static std::unique_ptr<primitive_impl> create(const moe_mask_gen_reshape_node& arg, const kernel_impl_params& impl_param) {
        return std::make_unique<moe_mask_gen_reshape_impl>();
    }
};

namespace detail {
attach_moe_mask_gen_impl::attach_moe_mask_gen_impl() {
    auto formats = {
        format::bfyx,
    };

    auto types = {
        data_types::i32,
        data_types::i64,
        data_types::f32,
    };

    implementation_map<moe_mask_gen>::add(impl_types::cpu, shape_types::static_shape, moe_mask_gen_impl::create, types, formats);
    implementation_map<moe_mask_gen>::add(impl_types::cpu, shape_types::dynamic_shape, moe_mask_gen_impl::create, types, formats);
}

attach_moe_mask_gen_reshape_impl::attach_moe_mask_gen_reshape_impl() {
    auto formats = {
        format::bfyx,
    };

    auto types = {
        data_types::i32,
        data_types::i64,
        data_types::f32,
    };

    implementation_map<moe_mask_gen_reshape>::add(impl_types::cpu, shape_types::static_shape, moe_mask_gen_reshape_impl::create, types, formats);
    implementation_map<moe_mask_gen_reshape>::add(impl_types::cpu, shape_types::dynamic_shape, moe_mask_gen_reshape_impl::create, types, formats);
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::moe_mask_gen_reshape_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_mask_gen_reshape)
