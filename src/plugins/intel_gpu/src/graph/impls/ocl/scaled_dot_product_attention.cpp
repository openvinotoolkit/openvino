// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi_stage_primitive.hpp"
#include "scaled_dot_product_attention_inst.h"
#include "kv_cache_inst.h"

#include "sdpa/sdpa_kernel_selector.h"
#include "sdpa/sdpa_kernel_base.h"

namespace cldnn {
namespace ocl {

// SDPA impl may create 2 versions of the kernel internally
// 1. Default SDPA kernels
// 2. SDPA kernels with indirect access to one of the inputs
// This feature is used to avoid perf drop when we create single kernel which checks batch size in runtime
// Can be reverted once performance of the kernel is improved
struct scaled_dot_product_attention_impl : multi_stage_primitive<scaled_dot_product_attention> {
    using parent = multi_stage_primitive<scaled_dot_product_attention>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::sdpa_kernel_selector;
    using kernel_params_t = kernel_selector::sdpa_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::scaled_dot_product_attention_impl)

    const uint32_t default_sdpa = 0;
    const uint32_t indirect_sdpa = 1;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<scaled_dot_product_attention_impl>(*this);
    }

    scaled_dot_product_attention_impl() = default;

    scaled_dot_product_attention_impl(const std::vector<kernel_selector::kernel_data>& kd) : parent(kd) {
        this->can_reuse_memory = true;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernels_data[default_sdpa].kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[default_sdpa]);
            if (_kernels_data.size() == 2) {
                auto bt_kernel_impl = kernel_selector.GetImplementation(_kernels_data[indirect_sdpa].kernelName);
                bt_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[indirect_sdpa]);
            }
        }
    }

protected:
    std::vector<layout> get_internal_buffer_layouts_impl(const kernel_impl_params& /*params*/) const override {
        // TODO: current implementation is supposed to have the same kernel version for both indirect/default paths,
        // considering this, we may assume that both indirect/default kernels have absolutely the same intermediate
        // buffers number and its' sizes (since update_dispatch_data is called for both kernels too), and
        // do not double memory allocations during reallocate_if_needed() function call
        std::vector<layout> layouts;
        if (_kernels_data.size() > 0 && !_kernels_data[0].internalBufferSizes.empty()) {
            auto dtype = from_data_type(_kernels_data[0].internalBufferDataType);
            const auto bpp = data_type_traits::size_of(dtype);
            for (auto size : _kernels_data[0].internalBufferSizes) {
                layout inbuf_layout = {dtype, format::bfyx, // simple linear format (flattern to x channel)
                                        {1, 1, 1, (tensor::value_type)(size / bpp)}};
                layouts.push_back(inbuf_layout);
            }
        }

        return layouts;
    }

    static size_t get_beam_table_id(std::shared_ptr<const scaled_dot_product_attention> primitive) {
        return primitive->input_size() - 1;
    }

    static bool has_indirect_inputs(const kernel_impl_params& impl_param) {
        const auto& desc = impl_param.typed_desc<scaled_dot_product_attention>();
        return desc->indirect_axis != -1;
    }

    kernel_arguments_data get_arguments(const scaled_dot_product_attention_inst& instance, size_t stage) const override {
        kernel_arguments_data args;

        auto inputs_num = instance.inputs_memory_count();
        if (instance.has_indirect_inputs() && stage == default_sdpa)
            inputs_num--;

        for (size_t i = 0; i < inputs_num; i++) {
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

    void set_arguments_impl(scaled_dot_product_attention_inst& instance) override {}

    event::ptr execute_stage(const std::vector<event::ptr>& events, scaled_dot_product_attention_inst& instance, size_t stage) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;
        size_t kernel_offset = 0;

        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution)
                continue;

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the desc's users is CPU implementation or network's output, set desc as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;
            auto args = get_arguments(instance, stage);
            args.scalars = &params.scalars;

            for (size_t i = 0; i < instance.get_intermediates_memories().size(); i++)
                args.intermediates.push_back(instance.get_intermediates_memories()[i]);

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

        return aggregate_events(all_events, stream, all_events.size() > 1);
    }

    bool need_indirect_load(const scaled_dot_product_attention_inst& instance) const {
        auto desc = instance.get_typed_desc<scaled_dot_product_attention>();

        if (!instance.has_indirect_inputs())
            return false;

        const auto& params = *instance.get_impl_params();
        const auto indirect_axis = desc->indirect_axis;
        if (params.input_layouts[get_beam_table_id(desc)].get_partial_shape()[indirect_axis].get_length() == 1)
            return false;

        const auto& deps = instance.dependencies();

        const auto indirect_dep_idx = 1;
        const auto& indirect_dep = deps[indirect_dep_idx].first;
        if (dynamic_cast<const kv_cache_inst*>(indirect_dep) == nullptr) {
            return true;
        }

        auto state_layout = indirect_dep->get_impl_params()->get_input_layout(0);
        bool is_prefill = state_layout.count() == 0;
        return !is_prefill;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, scaled_dot_product_attention_inst& instance) override {
        if (need_indirect_load(instance))
            return execute_stage(events, instance, indirect_sdpa);
        else
            return execute_stage(events, instance, default_sdpa);
    }

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param) {
        kernel_selector::sdpa_configuration config;

        auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
            if (order.empty())
                return pshape;

            auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
            for (size_t i = 0; i < order.size(); i++) {
                transposed_pshape[i] = pshape[order[i]];
            }
            return transposed_pshape;
        };

        const auto& desc = impl_param.typed_desc<scaled_dot_product_attention>();
        const auto query_shape = transpose_pshape(impl_param.get_input_layout(0).get_partial_shape(), desc->input_q_transpose_order);
        const auto key_shape = transpose_pshape(impl_param.get_input_layout(1).get_partial_shape(), desc->input_k_transpose_order);
        const auto value_shape = transpose_pshape(impl_param.get_input_layout(2).get_partial_shape(), desc->input_v_transpose_order);

        OPENVINO_ASSERT(key_shape == value_shape, "[GPU] The shapes of key and value inputs are expected to be equal");
        for (size_t i = 0; i < query_shape.size(); ++i) {
            if (query_shape[i].is_static() && key_shape[i].is_static() && value_shape[i].is_static()) {
                if (query_shape[i].get_length() > key_shape[i].get_length()) {
                    config.broadcast_axis = desc->input_k_transpose_order[i];
                    config.group_size = query_shape[i].get_length() / key_shape[i].get_length();
                }
            }
        }

        if (query_shape[query_shape.size() - 1].is_static())
            config.head_size = query_shape[query_shape.size() - 1].get_length();

        config.is_causal = desc->is_causal;

        return config;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic, bool indirect = false) {
        const auto& desc = impl_param.typed_desc<scaled_dot_product_attention>();
        auto params = get_default_params<kernel_selector::sdpa_params>(impl_param, is_dynamic);

        auto data_inputs_num = impl_param.input_layouts.size();
        if (has_indirect_inputs(impl_param))
            data_inputs_num--;

        params.inputs.resize(data_inputs_num);
        for (size_t i = 0; i < data_inputs_num; i++) {
            params.inputs[i] = convert_data_tensor(impl_param.get_input_layout(i));
        }

        params.conf = get_sdpa_configuration(impl_param);

        params.input0_order = desc->input_q_transpose_order;
        params.input1_order = desc->input_k_transpose_order;
        params.input2_order = desc->input_v_transpose_order;
        params.output_order = desc->output_transpose_order;

        if (indirect && has_indirect_inputs(impl_param)) {
            params.beam_table = convert_data_tensor(impl_param.get_input_layout(get_beam_table_id(desc)));
            params.indirect_axis = desc->indirect_axis;
        }

        params.set_dynamic_shape_offsets();

        // Need to adjust sdpa kernel offset to consider beam table input
        if (has_indirect_inputs(impl_param)) {
            auto out_offset = params.outputs[0].get_dynamic_shape_offset();
            if (indirect)
                params.beam_table.SetDynamicShapeOffset(out_offset);

            params.outputs[0].SetDynamicShapeOffset(out_offset + kernel_selector::DataTensor::max_rank());
        }

        return params;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<scaled_dot_product_attention>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto sdpa_kernel_params = get_kernel_params(impl_param, impl_param.is_dynamic());
        auto& kernel_selector = kernel_selector_t::Instance();
        kernels_data.push_back(kernel_selector.get_best_kernel(sdpa_kernel_params));

        if (has_indirect_inputs(impl_param)) {
            auto indirect_kernel_params = get_kernel_params(impl_param, impl_param.is_dynamic(), true);
            kernels_data.push_back(kernel_selector.get_best_kernel(indirect_kernel_params));
        }

        return cldnn::make_unique<scaled_dot_product_attention_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernels_data[default_sdpa].params == nullptr) {
            _kernels_data[default_sdpa].params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }
        update_shapes(*_kernels_data[default_sdpa].params, impl_param);
        (_kernels_data[default_sdpa].update_dispatch_data_func)(*_kernels_data[default_sdpa].params, _kernels_data[default_sdpa]);

        if (_kernels_data.size() == 2) {
            if (_kernels_data[indirect_sdpa].params == nullptr) {
                _kernels_data[indirect_sdpa].params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
            }
            update_shapes(*_kernels_data[indirect_sdpa].params, impl_param);
            (_kernels_data[indirect_sdpa].update_dispatch_data_func)(*_kernels_data[indirect_sdpa].params, _kernels_data[indirect_sdpa]);
        }
    }
};

namespace detail {

attach_scaled_dot_product_attention_impl::attach_scaled_dot_product_attention_impl() {
    using sdpa_prim = scaled_dot_product_attention;

    auto types = {
        data_types::f32,
        data_types::f16,
    };

    auto formats = {
        format::bfyx,
    };

    implementation_map<sdpa_prim>::add(impl_types::ocl,
                                       shape_types::static_shape,
                                       scaled_dot_product_attention_impl::create,
                                       types,
                                       formats);

    implementation_map<sdpa_prim>::add(impl_types::ocl,
                                       shape_types::dynamic_shape,
                                       scaled_dot_product_attention_impl::create,
                                       types,
                                       formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::scaled_dot_product_attention_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scaled_dot_product_attention)
