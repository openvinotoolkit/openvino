// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "multi_stage_primitive.hpp"

#include "lora_inst.h"
#include "lora.hpp"
#include "lora/lora_kernel_selector.h"
#include "lora/lora_kernel_base.h"

namespace cldnn {
namespace ocl {

struct lora_impl : multi_stage_primitive<lora> {
    using parent = multi_stage_primitive<lora>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lora_kernel_selector;
    using kernel_params_t = kernel_selector::lora_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::lora_impl);

    const uint32_t optimized_kernel = 0;
    const uint32_t reference_kernel = 1;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<lora_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            for (auto& kd : _kernels_data) {
                if (kd.kernelName.length() != 0) {
                    auto kernel_impl = kernel_selector.GetImplementation(kd.kernelName);
                    kernel_impl->GetUpdateDispatchDataFunc(kd);
                }
            }
        }
    }

    kernel_arguments_data get_arguments(const lora_inst& instance, size_t stage) const override {
        kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back(instance.fused_memory(i));
            }
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();

        return args;
    }

    bool is_optimized_kernel_supported(const lora_inst& instance) {
        const auto& in_dtype = instance.get_input_layout().data_type;
        size_t subgroup_size = in_dtype == ov::element::f16 ? 16 : 8;

        const auto& state_a_layout = instance.get_input_layout(2);
        size_t input_state = state_a_layout.get_shape().back();
        if (input_state % subgroup_size != 0) {
            return false;
        }

        const auto& alpha_layout = instance.get_input_layout(3);
        size_t lora_rank = alpha_layout.get_shape().back();
        if (lora_rank % subgroup_size != 0) {
            return false;
        }

        const auto& state_b_layout = instance.get_input_layout(4);
        size_t output_state = state_b_layout.get_shape().front();
        if (output_state % subgroup_size != 0) {
            return false;
        }

        return true;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, lora_inst& instance) override {
        if (is_optimized_kernel_supported(instance)) {
            return execute_stage(events, instance, optimized_kernel);
        } else {
            return execute_stage(events, instance, reference_kernel);
        }
    }

    void set_arguments_impl(lora_inst& instance) override {}

    event::ptr execute_stage(const std::vector<event::ptr>& events, lora_inst& instance, size_t stage) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;
        size_t kernel_offset = 0;
        bool skip_full_lora = true;

        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution) {
                continue;
            } else {
                skip_full_lora = false;
            }

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the desc's users is CPU implementation or network's output, set desc as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;
            auto args = get_arguments(instance, stage);
            args.scalars = &params.scalars;

            if (stage == optimized_kernel) {
                for (const auto& m : instance.get_intermediates_memories()) {
                    args.intermediates.push_back(m);
                }
            }

            stream.set_arguments(*_kernels[idx_final], _kernels_data[stage].kernels[kd_idx].params, args);

            const auto& gws = params.workGroups.global;
            const auto& lws = params.workGroups.local;

            GPU_DEBUG_TRACE_DETAIL << "Enqueue stage " << stage << " kernel " << idx_final << ": gws=[" << gws[0] << ", " << gws[1] << ", " << gws[2] << "] "
                                   << "lws=[" << lws[0] << ", " << lws[1] << ", " << lws[2] << "]"
                                   << (needs_completion_event ? " has_completion_event=true" : "") << std::endl;

            auto ev = stream.enqueue_kernel(*_kernels[idx_final], params, args, tmp_events, needs_completion_event);
            if (_kernels_data[stage].needs_sub_kernels_sync) {
                tmp_events = {ev};
            }
            all_events.push_back(ev);
        }

        if (skip_full_lora) {
            for (auto& ev : events) {
                all_events.push_back(ev);
            }
        }

        return stream.aggregate_events(all_events, all_events.size() > 1);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false, bool is_ref_kernel = false) {
        auto params = get_default_params<kernel_selector::lora_params>(impl_param, is_shape_agnostic);

        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        size_t fused_dep_size = 0;
        for (const auto& fused_op : params.fused_ops) {
            fused_dep_size += fused_op.dep_size;
        }
        params.lora_count = (params.inputs.size() - fused_dep_size - 2ul) / 3ul;
        params.is_ref_kernel = is_ref_kernel;
        params.set_dynamic_shape_offsets();

        return params;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<lora>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto& kernel_selector = kernel_selector_t::Instance();
        auto canonicalized_params = static_canonicalize_shapes(impl_param);

        auto optimized_kernel_params = get_kernel_params(canonicalized_params, impl_param.is_dynamic());
        kernels_data.push_back(kernel_selector.get_best_kernel(optimized_kernel_params));

        auto reference_kernel_params = get_kernel_params(canonicalized_params, impl_param.is_dynamic(), true);
        kernels_data.push_back(kernel_selector.get_best_kernel(reference_kernel_params));

        return std::make_unique<lora_impl>(kernels_data);
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = impl_params;
        for (size_t i = 0; i < 2; ++i) {
            if (impl_params.get_input_layout(i).get_partial_shape().size() == 2) {
                auto input_pshape = impl_params.input_layouts[i].get_partial_shape();
                input_pshape.insert(input_pshape.begin(), 1);
                updated_impl_params.input_layouts[i].set_partial_shape(input_pshape);
            }
        }
        return primitive_impl::static_canonicalize_shapes(updated_impl_params);
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        for (size_t kd_idx = 0; kd_idx < _kernels_data.size(); ++kd_idx) {
            auto& kd = _kernels_data[kd_idx];
            // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
            if (kd.params == nullptr) {
                bool is_ref_kernel = static_cast<bool>(kd_idx);
                kd.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true, is_ref_kernel));
            }

            update_shapes(*kd.params, impl_param);
            (kd.update_dispatch_data_func)(*kd.params, kd);
        }
    }
};

std::unique_ptr<primitive_impl> LoraImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<lora>());
    return cldnn::ocl::lora_impl::create(static_cast<const lora_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lora_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lora)
