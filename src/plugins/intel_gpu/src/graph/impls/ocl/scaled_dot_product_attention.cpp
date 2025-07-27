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
        return make_deep_copy<scaled_dot_product_attention_impl, kernel_params_t>(*this);
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
            if (_kernels_data.size() >= 2) {
                auto bt_kernel_impl = kernel_selector.GetImplementation(_kernels_data[indirect_sdpa].kernelName);
                bt_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[indirect_sdpa]);
            }
            if (_kernels_data.size() == 3) {
                auto bt_kernel_impl = kernel_selector.GetImplementation(_kernels_data[2].kernelName);
                bt_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[2]);
            }
        }
    }

protected:
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params&) const override {
        // Look for the first sdpa_opt kernel entry. Currently, it can be used as default sdpa, indirect sdpa, or for both default
        // and indirect cases. All of sdpa_opt kernels use the same internal buffers, so we can find the first sdpa_opt and
        // use its` internal buffers configuration. The following scenarios are possible:
        // 1) _kernels_data[0] - micro_sdpa (default)
        //   => internal buffers are not needed
        // 2) _kernels_data[0] - sdpa_opt (default)
        //   => use internal buffers from [0] kernel
        // 2) _kernels_data[0] - sdpa_opt (default)
        //    _kernels_data[1] - sdpa_opt (indirect)
        //   => use internal buffers from [0] kernel
        // 3) _kernels_data[0] - micro_sdpa (default)
        //    _kernels_data[1] - sdpa_opt (indirect)
        //   => use internal buffers from [1] kernel
        size_t kernel_idx = _kernels_data.size();
        if (_kernels_data.size() >= 1 && !_kernels_data[0].internalBuffers.empty()) {
            kernel_idx = 0;
        } else if (_kernels_data.size() >= 2 && !_kernels_data[1].internalBuffers.empty()) {
            kernel_idx = 1;
        }

        std::vector<BufferDescriptor> internal_buffers;
        if (kernel_idx < _kernels_data.size()) {
            auto dtype = from_data_type(_kernels_data[kernel_idx].internalBufferDataType);
            const auto bpp = data_type_traits::size_of(dtype);
            for (const auto& buffer : _kernels_data[kernel_idx].internalBuffers) {
                internal_buffers.emplace_back(buffer.byte_count / bpp, dtype, buffer.lockable);
            }
        }

        return internal_buffers;
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
        const auto desc = instance.get_node().as<scaled_dot_product_attention>().get_primitive();

        auto inputs_num = instance.inputs_memory_count();
        if (instance.has_indirect_inputs() && stage == default_sdpa)
            inputs_num--;

        const size_t attn_mask_idx = 3;
        const size_t scale_idx = 4;
        for (size_t i = 0; i < inputs_num; i++) {
            if (i == attn_mask_idx && desc->attn_mask_val.has_value()) {
                continue;
            }
            if (i == scale_idx && desc->scale_val.has_value()) {
                continue;
            }
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

        return stream.aggregate_events(all_events, all_events.size() > 1);
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

    bool need_sdpa_opt_load(const scaled_dot_product_attention_inst& instance) const {
        if (_kernels_data.size() < 2)
            return false;

        if (instance.has_indirect_inputs() && _kernels_data.size() < 3)
            return false;

        const auto& query_layout = instance.get_impl_params()->get_input_layout(0);

        auto get_reordered_dimension = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order, size_t idx) -> const ov::Dimension& {
            if (order.empty())
                return pshape[idx];

            return pshape[order[idx]];
        };

        const auto& desc = instance.get_impl_params()->typed_desc<scaled_dot_product_attention>();
        const auto dim_L = get_reordered_dimension(query_layout.get_partial_shape(), desc->input_q_transpose_order, 2 /* y */);

        bool is_generate = dim_L.get_length() == 1;  // L
        return is_generate;
    }

    static bool requires_shape_canonicalization(const kernel_impl_params& impl_params) {
        auto extend_output = impl_params.output_layouts[0].get_partial_shape().size() < 4;
        auto extend_attn_mask = false;

        // According to SDPA specification, attention mask should have 2-dimensions or more or empty
        const auto attn_mask_idx = 3;
        if (impl_params.input_layouts.size() > attn_mask_idx) {
            const auto& attn_mask_shape = impl_params.get_input_layout(attn_mask_idx).get_partial_shape();
            extend_attn_mask = attn_mask_shape.size() != 0 && attn_mask_shape.size() < 4;
        }

        return extend_output || extend_attn_mask;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, scaled_dot_product_attention_inst& instance) override {
        if (need_indirect_load(instance)) {
            return execute_stage(events, instance, indirect_sdpa);
        } else if (need_sdpa_opt_load(instance)) {
            return execute_stage(events, instance, _kernels_data.size() -1 /* the last */);
        } else {
            return execute_stage(events, instance, default_sdpa);
        }
    }

    static kernel_selector::sdpa_configuration get_sdpa_configuration(const kernel_impl_params& impl_param,
                                                                      const std::vector<int64_t>& input_q_transpose_order,
                                                                      const std::vector<int64_t>& input_k_transpose_order,
                                                                      const std::vector<int64_t>& input_v_transpose_order) {
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
        const auto query_shape = transpose_pshape(impl_param.get_input_layout(0).get_partial_shape(), input_q_transpose_order);
        const auto key_shape = transpose_pshape(impl_param.get_input_layout(1).get_partial_shape(), input_k_transpose_order);
        const auto value_shape = transpose_pshape(impl_param.get_input_layout(2).get_partial_shape(), input_v_transpose_order);

        const auto num_heads_dim = 1;
        if (query_shape[num_heads_dim].is_static() && key_shape[num_heads_dim].is_static() && value_shape[num_heads_dim].is_static()) {
            if (query_shape[num_heads_dim].get_length() > key_shape[num_heads_dim].get_length()) {
                config.broadcast_axis = desc->input_k_transpose_order[num_heads_dim];
                config.kv_group_size = query_shape[num_heads_dim].get_length() / key_shape[num_heads_dim].get_length();
            }
        }

        if (query_shape[query_shape.size() - 1].is_static())
            config.k_head_size = query_shape[query_shape.size() - 1].get_length();

        if (value_shape[value_shape.size() - 1].is_static())
            config.v_head_size = value_shape[value_shape.size() - 1].get_length();

        config.is_causal = desc->is_causal;

        if (desc->scale_val.has_value()) {
            config.has_const_scale_val = true;
            config.scale_val = desc->scale_val.value();
        } else {
            config.has_const_scale_val = false;
        }

        if (desc->attn_mask_val.has_value()) {
            config.has_const_attn_mask_val = true;
            config.attn_mask_val = desc->attn_mask_val.value();
        } else {
            config.has_const_attn_mask_val = false;
        }

        if (desc->is_kv_compressed) {
            const auto& group_sizes = desc->quantization_attributes.group_sizes;
            const auto non_compressed_dims = std::count(group_sizes.begin(), group_sizes.end(), 1);

            config.per_head_quantization = (group_sizes.size() - non_compressed_dims) == 1;
            config.is_kv_compressed = desc->is_kv_compressed;
            config.use_asymmetric_quantization =
                desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
            config.combine_scales_and_zp =
                desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        }

        return config;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_dynamic, bool indirect = false) {
        const auto& desc = impl_param.typed_desc<scaled_dot_product_attention>();
        auto params = get_default_params<kernel_selector::sdpa_params>(impl_param, is_dynamic);

        auto data_inputs_num = impl_param.input_layouts.size();
        if (has_indirect_inputs(impl_param))
            data_inputs_num--;

        if (desc->scale_val.has_value()) {
            data_inputs_num--;
        }
        if (desc->attn_mask_val.has_value()) {
            data_inputs_num--;
        }

        auto has_zp_input_buffers = desc->get_compression_zp_inputs_num() > 0;
        if (desc->is_kv_compressed) {
            data_inputs_num -= 2; // key and value compression scales are handled separately

            if (desc->get_compression_zp_inputs_num() > 0)
                data_inputs_num -= 2; // key and value compression zp are handled separately
        }

        params.inputs.resize(data_inputs_num);
        for (size_t i = 0; i < data_inputs_num; i++) {
            params.inputs[i] = convert_data_tensor(impl_param.get_input_layout(i));
        }

        if (desc->scale_val.has_value()) {
            data_inputs_num++;
        }
        if (desc->attn_mask_val.has_value()) {
            data_inputs_num++;
        }

        auto extend_order_in_num_heads_dim = [](const std::vector<int64_t>& order, size_t rank = 4) {
            if (order.size() == rank) {
                return order;
            }

            std::vector<int64_t> extended_order(rank, 0);
            const size_t num_heads_dim = 1;
            // For 3D dimension, extend it to 4D by adding 1 for num_heads_dim
            for (size_t i = 0, j = 0; i < rank; ++i) {
                if (i == num_heads_dim) {
                    extended_order[num_heads_dim] = 1;
                } else {
                    extended_order[i] = (static_cast<size_t>(order[j]) < num_heads_dim) ? order[j] : order[j] + 1;
                    j++;
                }
            }
            return extended_order;
        };
        auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
        auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
        auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
        auto extended_output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);

        params.conf = get_sdpa_configuration(impl_param,
                                             extended_input_q_transpose_order,
                                             extended_input_k_transpose_order,
                                             extended_input_v_transpose_order);

        params.input0_order = extended_input_q_transpose_order;
        params.input1_order = extended_input_k_transpose_order;
        params.input2_order = extended_input_v_transpose_order;
        params.output_order = extended_output_transpose_order;

        if (indirect && has_indirect_inputs(impl_param)) {
            params.beam_table = convert_data_tensor(impl_param.get_input_layout(get_beam_table_id(desc)));
            params.indirect_axis = desc->indirect_axis;
        }

        if (desc->is_kv_compressed) {
            params.key_cache_comp_scale = convert_data_tensor(impl_param.get_input_layout(data_inputs_num));
            params.value_cache_comp_scale = convert_data_tensor(impl_param.get_input_layout(data_inputs_num + 1));

            if (has_zp_input_buffers) {
                params.key_cache_comp_zp = convert_data_tensor(impl_param.get_input_layout(data_inputs_num + 2));
                params.value_cache_comp_zp = convert_data_tensor(impl_param.get_input_layout(data_inputs_num + 3));
            }
        }

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        std::map<size_t, size_t> in_tensor_to_offset_map;
        for (size_t i = 0; i < data_inputs_num; i++) {
            in_tensor_to_offset_map[i] = in_offsets_map.at(i);
        }

        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        if (desc->is_kv_compressed) {
            params.key_cache_comp_scale.SetDynamicShapeOffset(in_offsets_map.at(data_inputs_num));
            params.value_cache_comp_scale.SetDynamicShapeOffset(in_offsets_map.at(data_inputs_num + 1));

            if (has_zp_input_buffers) {
                params.key_cache_comp_zp.SetDynamicShapeOffset(in_offsets_map.at(data_inputs_num + 2));
                params.value_cache_comp_zp.SetDynamicShapeOffset(in_offsets_map.at(data_inputs_num + 3));
            }
        }

        if (indirect && has_indirect_inputs(impl_param)) {
            params.beam_table.SetDynamicShapeOffset(in_offsets_map.at(get_beam_table_id(desc)));
        }

        params.could_use_flashattn_v2 = impl_param.get_program().get_config().get_could_use_flashattn_v2();

        return params;
    }

    static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
        auto updated_impl_params = impl_params;

        auto extend_pshape_to_rank_in_num_heads_dim = [](ov::PartialShape pshape, size_t rank = 4) {
            if (pshape.size() == rank) {
                return pshape;
            }
            const size_t num_heads_dim = 1;
            pshape.insert(pshape.begin() + num_heads_dim, ov::Dimension(1));
            return pshape;
        };

        const auto attn_mask_idx = 3;
        if (updated_impl_params.input_layouts.size() > attn_mask_idx) {
            const auto attn_mask_shape = updated_impl_params.input_layouts[attn_mask_idx].get_partial_shape();
            updated_impl_params.input_layouts[attn_mask_idx].set_partial_shape(extend_shape_to_rank_from_begin(attn_mask_shape));
        }

        // For scale of 1D tensor or attention_mask of empty shape, use extend_shape_to_rank_from_end as before
        for (auto& input_layout : updated_impl_params.input_layouts) {
            input_layout.set_partial_shape(input_layout.get_partial_shape().size() <= 1 ?
                                           extend_shape_to_rank_from_end(input_layout.get_partial_shape()) :
                                           extend_pshape_to_rank_in_num_heads_dim(input_layout.get_partial_shape()));
        }

        auto& output_layout = updated_impl_params.output_layouts[0];
        output_layout.set_partial_shape(extend_pshape_to_rank_in_num_heads_dim(output_layout.get_partial_shape()));

        return updated_impl_params;
    }

    kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
        return static_canonicalize_shapes(impl_params);
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<scaled_dot_product_attention>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto& kernel_selector = kernel_selector_t::Instance();
        const bool is_output_rank_4d = impl_param.output_layouts[0].get_partial_shape().size() == 4;
        auto params = requires_shape_canonicalization(impl_param) ? static_canonicalize_shapes(impl_param) : impl_param;

        auto sdpa_kernel_params = get_kernel_params(params, params.is_dynamic());
        // Known limitation: In vision encoding model of qwen-vl, when the shape of sdpa is 3D and num_heads is 1,
        // there is an accuracy issue with sdpa_micro kernel. Therefore, it is currently restricted to execute with sdpa_opt kernel.
        if (!is_output_rank_4d)
            sdpa_kernel_params.should_use_sdpa_opt = true;
        kernels_data.push_back(kernel_selector.get_best_kernel(sdpa_kernel_params));
        if (has_indirect_inputs(params)) {
            auto indirect_kernel_params = get_kernel_params(params, params.is_dynamic(), true);
            kernels_data.push_back(kernel_selector.get_best_kernel(indirect_kernel_params));
        }

        const auto& gfx_ver = params.get_program().get_engine().get_device_info().gfx_ver;
        if (gfx_ver.major == 12 && gfx_ver.minor == 74) { // ARL only
            sdpa_kernel_params.should_use_sdpa_opt = true;
            kernels_data.push_back(kernel_selector.get_best_kernel(sdpa_kernel_params));
        }

        return std::make_unique<scaled_dot_product_attention_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernels_data[default_sdpa].params == nullptr) {
            _kernels_data[default_sdpa].params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }
        update_shapes(*_kernels_data[default_sdpa].params, impl_param);
        (_kernels_data[default_sdpa].update_dispatch_data_func)(*_kernels_data[default_sdpa].params, _kernels_data[default_sdpa]);

        if (_kernels_data.size() >= 2) {
            if (_kernels_data[indirect_sdpa].params == nullptr) {
                _kernels_data[indirect_sdpa].params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
            }
            update_shapes(*_kernels_data[indirect_sdpa].params, impl_param);
            (_kernels_data[indirect_sdpa].update_dispatch_data_func)(*_kernels_data[indirect_sdpa].params, _kernels_data[indirect_sdpa]);
        }
        if (_kernels_data.size() == 3) {
            (_kernels_data[2].update_dispatch_data_func)(*_kernels_data[default_sdpa].params, _kernels_data[2]);
        }
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        auto new_impl_params = requires_shape_canonicalization(impl_params) ? canonicalize_shapes(impl_params) : impl_params;
        update_dispatch_data(new_impl_params);
        inst.update_shape_info_tensor(new_impl_params);
    }
};

namespace detail {

attach_scaled_dot_product_attention_impl::attach_scaled_dot_product_attention_impl() {
    using sdpa_prim = scaled_dot_product_attention;

    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
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
