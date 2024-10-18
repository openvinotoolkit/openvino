// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "multi_stage_primitive.hpp"

#include "kv_cache_inst.h"
#include "dynamic_quantize_inst.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "concatenation/concatenation_kernel_base.h"
#include "beam_table_update/beam_table_update_kernel_selector.hpp"
#include "beam_table_update/beam_table_update_kernel_ref.hpp"
#include "dynamic_quantize/dynamic_quantize_kernel_selector.h"
#include "dynamic_quantize/dynamic_quantize_kernel_kv_cache.h"
#include "openvino/core/dimension.hpp"

#include <limits.h>

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::concat_axis convert_axis(int64_t axis, size_t rank) {
    auto cldnn_axis = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    if (cldnn_axis >= static_cast<int64_t>(rank))
        OPENVINO_THROW("kv_cache axis exceeds number of dimensions");

    // Difference in dimension ordering between OV and GPU plugin,
    // reverse spatial dimensions after batch and feature.
    if (cldnn_axis >= 2) {
        auto spatial_axis = cldnn_axis - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max<size_t>(rank, 4) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0: return kernel_selector::concat_axis::BATCH;
        case 1: return kernel_selector::concat_axis::FEATURE;
        case 2: return kernel_selector::concat_axis::X;
        case 3: return kernel_selector::concat_axis::Y;
        case 4: return kernel_selector::concat_axis::Z;
        case 5: return kernel_selector::concat_axis::W;
        default: OPENVINO_THROW("Unsupported kv_cache axis: ", axis);
    }

    return kernel_selector::concat_axis::FEATURE;  // shouldn't get here
}

}  // namespace

struct kv_cache_impl : multi_stage_primitive<kv_cache> {
    using parent = multi_stage_primitive<kv_cache>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::concatenation_kernel_selector;
    using kernel_params_t = kernel_selector::concatenation_params;

    using bt_kernel_selector_t = kernel_selector::beam_table_update_kernel_selector;
    using bt_kernel_params_t = kernel_selector::beam_table_update_params;

    using dq_kernel_selector_t = kernel_selector::dynamic_quantize_kernel_selector;
    using dq_kernel_params_t = kernel_selector::dynamic_quantize_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::kv_cache_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<kv_cache_impl>(*this);
    }

    const size_t concat_stage = 0;
    const size_t beam_table_stage = 1;
    const size_t dq_stage = 2;
    const size_t scale_concat_stage = 3;
    const size_t zp_concat_stage = 4;

    cldnn::memory::ptr beam_table_prev = nullptr;
    cldnn::memory::ptr beam_table_new = nullptr;

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernels_data[concat_stage].kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[concat_stage]);
            if (_kernels_data.size() >= 2) {
                auto& bt_kernel_selector = bt_kernel_selector_t::Instance();
                auto bt_kernel_impl = bt_kernel_selector.GetImplementation(_kernels_data[beam_table_stage].kernelName);
                bt_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[beam_table_stage]);
            }

            if (_kernels_data.size() >= 3) {
                auto& dq_kernel_selector = dq_kernel_selector_t::Instance();
                auto dq_kernel_impl = dq_kernel_selector.GetImplementation(_kernels_data[dq_stage].kernelName);
                dq_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[dq_stage]);
            }

            if (_kernels_data.size() >= 4) {
                auto& scale_zp_concat_kernel_selector = kernel_selector_t::Instance();
                auto scale_zp_concat_kernel_impl = scale_zp_concat_kernel_selector.GetImplementation(_kernels_data[scale_concat_stage].kernelName);
                scale_zp_concat_kernel_impl->GetUpdateDispatchDataFunc(_kernels_data[scale_concat_stage]);
            }
        }
    }
    void set_arguments_impl(kv_cache_inst& instance) override {}

    kernel_arguments_data get_arguments(const kv_cache_inst& instance, size_t stage) const override {
        // input buffers order: [past, new, (beam_table), (past_scale), (past_zp)]
        // output buffers order: [current, (beam_table), (current_scale), (current_zp)]
        kernel_arguments_data args;
        args.shape_info = instance.shape_info_memory_ptr();
        if (stage == concat_stage) {
            args.inputs = { instance.input_memory_ptr(0), instance.input_memory_ptr(1) };
            args.outputs = { instance.output_memory_ptr(0) };
        } else if (stage == beam_table_stage) {
            args.inputs = { beam_table_prev, instance.input_memory_ptr(2) };
            args.outputs = { beam_table_new };
        } else if (stage == dq_stage) {
            args.inputs = { instance.input_memory_ptr(1) };
            args.outputs = { instance.output_memory_ptr(0) };
            for (size_t i = 2; i < instance.outputs_memory_count(); i++) {
                args.outputs.push_back(instance.output_memory_ptr(i));
            }
        } else if (stage == scale_concat_stage) {
            args.inputs = { instance.input_memory_ptr(3) };
            args.outputs = { instance.output_memory_ptr(2) };
        } else if (stage == zp_concat_stage) {
            args.inputs = { instance.input_memory_ptr(4) };
            args.outputs = { instance.output_memory_ptr(3) };
        }
        return args;
    }

    void execute_stage(const std::vector<event::ptr>& events,
                       kv_cache_inst& instance,
                       std::vector<event::ptr>& all_events,
                       size_t stage,
                       size_t arguments_set = SIZE_MAX) {
        stream& stream = instance.get_network().get_stream();
        std::vector<event::ptr> tmp_events(events);
        size_t kernel_offset = 0;
        for (size_t s = 0; s < stage; s++) {
            kernel_offset += _kernels_data[s].kernels.size();
        }
        for (size_t kd_idx = 0; kd_idx < _kernels_data[stage].kernels.size(); ++kd_idx) {
            if (_kernels_data[stage].kernels[kd_idx].skip_execution)
                continue;

            size_t idx_final = kernel_offset + kd_idx;
            // If any user of the prim's users is CPU implementation or network's output, set prim as a output event (event won't be nullptr)
            bool needs_completion_event = instance.needs_completion_event();

            auto& params = _kernels_data[stage].kernels[kd_idx].params;
            auto custom_arguments = arguments_set == SIZE_MAX ? stage : arguments_set;
            auto args = get_arguments(instance, custom_arguments);
            args.scalars = &params.scalars;

            for (const auto& m : instance.get_intermediates_memories()) {
                args.intermediates.push_back(m);
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
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, kv_cache_inst& instance) override {
        const bool can_be_optimized = instance.get_impl_params()->_can_be_optimized;
        auto& stream = instance.get_network().get_stream();
        auto& engine = instance.get_network().get_engine();
        const auto& desc = instance.get_typed_desc<kv_cache>();
        auto& variable = instance.get_network().get_variable(desc->variable_info.variable_id);
        std::vector<event::ptr> res_events;

        execute_stage(events, instance, res_events, concat_stage);

        const auto& impl_param = *instance.get_impl_params();
        const auto& kv_in_shape = impl_param.input_layouts[0].get_partial_shape();
        const auto& kv_out_shape = impl_param.output_layouts[0].get_partial_shape();
        if (desc->indirect && ((kv_out_shape[desc->gather_axis].get_length() > 1) ||
                               (kv_in_shape[desc->concat_axis].get_length() == 0))) {
            const auto bt_alloc_type = engine.get_preferred_memory_allocation_type(false);
            auto beam_table_state =
                dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCache&>(variable).get_beam_table_state();
            const auto& bt_layout = instance.get_impl_params()->output_layouts[1];
            auto bt_shape = bt_layout.get_shape();
            std::swap(beam_table_prev, beam_table_new);

            if (!beam_table_new || beam_table_new->count() < ov::shape_size(bt_shape)) {
                bt_shape[desc->concat_axis] += instance.get_prealloc_iter_num();
                const layout bt_alloc_layout = {bt_shape, bt_layout.data_type, bt_layout.format};
                GPU_DEBUG_TRACE_DETAIL << "Realloc beam table to " << bt_alloc_layout.to_short_string() << std::endl;
                beam_table_new = engine.allocate_memory(bt_alloc_layout, bt_alloc_type, false);

                // Alloc prev mem too as it will be needed in the future
                // That also simplifies arguments setting a little bit as we don't need to handle an optional past state
                if (!beam_table_prev) {
                    beam_table_prev = engine.allocate_memory(bt_alloc_layout, bt_alloc_type, false);
                }
            }

            instance.set_output_memory(beam_table_new, false, 1);
            beam_table_state->set_memory(beam_table_new, instance.get_impl_params()->output_layouts[1]);

            auto bt_kernel_params = get_bt_update_kernel_params(impl_param, beam_table_state->is_set());
            (_kernels_data[beam_table_stage].update_dispatch_data_func)(bt_kernel_params, _kernels_data[beam_table_stage]);

            execute_stage(events, instance, res_events, beam_table_stage);
            beam_table_state->set();
        }

        if (desc->compressed) {
            // Copy scales to the new buffer if needed
            execute_stage(events, instance, res_events, scale_concat_stage, scale_concat_stage);

            if (desc->get_compression_zp_inputs_num() > 0) {
                // Copy zero points to the new buffer if needed
                execute_stage(events, instance, res_events, scale_concat_stage, zp_concat_stage);
            }

            // Perform dynamic quantization of new token data and append result to the KV-cache
            auto dq_params = get_dq_update_kernel_params(impl_param, impl_param.is_dynamic());
            (_kernels_data[dq_stage].update_dispatch_data_func)(dq_params, _kernels_data[dq_stage]);
            execute_stage(events, instance, res_events, dq_stage);

            auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable);
            compressed_cache_variable->get_compression_scale_state()->set();

            if (desc->get_compression_zp_inputs_num() > 0) {
                compressed_cache_variable->get_compression_zp_state()->set();
            }
        }

        variable.set();
        if (can_be_optimized) {
            GPU_DEBUG_TRACE_DETAIL << desc->id  << " : Output is same as variable memory! Skip copying " << std::endl;
            // When primitive is optimized, concat kernel writes directly to variable memory
            return stream.aggregate_events(res_events, res_events.size() > 1);
        } else {
            // Otherwise, we need to copy result from out buffer to state memory
            GPU_DEBUG_TRACE_DETAIL << desc->id  << " : Copying output to variable memory" << std::endl;

            stream.enqueue_barrier();

            std::vector<event::ptr> res_events;
            auto out = instance.get_network().get_engine().reinterpret_buffer(instance.output_memory(0), variable.get_memory()->get_layout());
            res_events.push_back(variable.get_memory()->copy_from(stream, *out, false));

            if (desc->compressed) {
                auto compressed_cache_variable = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&variable);

                auto scale_state = compressed_cache_variable->get_compression_scale_state();
                auto out_scale_mem = instance.get_network().get_engine().reinterpret_buffer(instance.output_memory(2), scale_state->get_memory()->get_layout());
                res_events.push_back(scale_state->get_memory()->copy_from(stream, *out_scale_mem, false));

                if (desc->get_compression_zp_inputs_num() > 0) {
                    auto zp_state = compressed_cache_variable->get_compression_zp_state();
                    auto out_zp_mem = instance.get_network().get_engine().reinterpret_buffer(instance.output_memory(3), zp_state->get_memory()->get_layout());
                    res_events.push_back(zp_state->get_memory()->copy_from(stream, *out_zp_mem, false));
                }
            }

            return stream.aggregate_events(res_events, res_events.size() > 1);
        }
    }

    static layout get_beam_table_layout(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto kv_layout = impl_param.get_input_layout(0);

        // expected to be normalized already on primitive creation
        auto concat_axis = primitive->concat_axis;
        auto gather_axis = primitive->gather_axis;

        auto kv_shape = kv_layout.get_partial_shape();
        auto beam_table_shape = ov::PartialShape(std::vector<size_t>(kv_shape.size(), 1));
        beam_table_shape[gather_axis] = kv_shape[gather_axis];
        beam_table_shape[concat_axis] = kv_shape[concat_axis];
        return layout{beam_table_shape, impl_param.output_layouts[1].data_type, format::get_default_format(beam_table_shape.size())};
    }

    static kernel_params_t get_concat_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto params = get_default_params<kernel_selector::concatenation_params>(impl_param, is_shape_agnostic);
        auto axis = primitive->concat_axis;

        const auto inputs_count = 2;
        params.inputs.resize(inputs_count);
        for (size_t i = 0; i < inputs_count; ++i) {
            params.inputs[i] = convert_data_tensor(impl_param.input_layouts[i]);
        }

        params.axis = convert_axis(axis, impl_param.get_output_layout().get_rank());
        params.kernelPerInput = true;

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset; // [kv_past, kv_new_token, [beam_idx, beam_table_past]
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset; // [kv_present, beam_table_present]
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(0)},
            {1, in_offsets_map.at(1)},
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)},
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static bt_kernel_params_t get_bt_update_kernel_params(const kernel_impl_params& impl_param, bool is_state_set = false) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto params = get_default_params<kernel_selector::beam_table_update_params>(impl_param, true);
        auto indirect_axis = primitive->gather_axis;

        auto inputs_count = 2;
        auto bt_present_layout = impl_param.output_layouts[1];
        auto bt_shape = extend_shape_to_rank_from_end(bt_present_layout.get_partial_shape(), 1);
        bt_present_layout.set_partial_shape(bt_shape);
        layout bt_past_layout = get_beam_table_layout(impl_param);

        auto beam_idx_l = impl_param.input_layouts[2];
        beam_idx_l.set_partial_shape(extend_shape_to_rank_from_end(beam_idx_l.get_partial_shape(), 4));

        params.inputs.resize(inputs_count);
        params.inputs[0] = convert_data_tensor(bt_past_layout);
        params.inputs[1] = convert_data_tensor(beam_idx_l);
        params.outputs[0] = convert_data_tensor(bt_present_layout);
        params.inputs.resize(inputs_count);
        params.is_state_set = is_state_set;
        params.indirect_axis = indirect_axis;

        const auto& desc = impl_param.typed_desc<kv_cache>();
        const auto compression_inputs = desc->get_compression_scales_inputs_num() + desc->get_compression_zp_inputs_num();
        const auto beam_table_past_idx = 3 + compression_inputs;

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset; // [kv_past, kv_new_token, [beam_idx, [scale_past], [zp_past], beam_table_past]]
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset; // [kv_present, beam_table_present, compression_scale_present]
        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(beam_table_past_idx)}, // beam_table_past
            {1, in_offsets_map.at(2)}, // beam_idx
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(1)}, // beam_table_present
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static dq_kernel_params_t get_dq_update_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto params = get_default_params<dq_kernel_params_t>(impl_param, is_shape_agnostic);

        params.append_axis = primitive->concat_axis;
        params.group_sizes = primitive->quantization_config.group_sizes;
        params.scales_output_order = primitive->scales_zp_output_order;
        params.use_asymmetric_quantization = primitive->quantization_config.is_asymmetric_quantization();
        params.combine_scales_and_zp = primitive->combine_scales_and_zp;

        const auto& past_kv_cache_shape = impl_param.input_layouts[0].get_partial_shape();
        params.axis_offset = past_kv_cache_shape[primitive->concat_axis].is_static() ? past_kv_cache_shape[primitive->concat_axis].get_length() : 0;

        auto inputs_count = 1;
        auto outputs_count = 2;
        params.inputs.resize(inputs_count);
        params.outputs.resize(outputs_count);

        auto current_token_layout = impl_param.input_layouts[1];
        auto present_layout = impl_param.output_layouts[0];
        auto present_scales_layout = impl_param.output_layouts[2];
        params.inputs[0] = convert_data_tensor(current_token_layout);
        params.outputs[0] = convert_data_tensor(present_layout);
        params.outputs[1] = convert_data_tensor(present_scales_layout);

        const bool has_zp_output_buffer = primitive->get_compression_zp_inputs_num() > 0;
        if (has_zp_output_buffer) {
            auto present_zp_layout = impl_param.output_layouts[3];
            params.outputs.resize(outputs_count + 1);
            params.outputs[2] = convert_data_tensor(present_zp_layout);
        }

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;

        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(1)}, // kv_new_token
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(0)}, // compressed_kv_present
            {1, out_offsets_map.at(2)}, // compression_scale_present
        };

        if (has_zp_output_buffer) {
            out_tensor_to_offset_map.emplace(2, out_offsets_map.at(3)); // compression_zp_present
        }
        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static kernel_params_t get_compression_scale_update_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::concatenation_params>(impl_param, is_shape_agnostic);

        const auto concat_axis = 2;
        params.axis = convert_axis(concat_axis, impl_param.get_output_layout().get_rank());

        auto inputs_count = 1;
        auto comp_scale_past_layout = impl_param.input_layouts[3];
        auto comp_scale_present_layout = impl_param.output_layouts[2];

        params.inputs.resize(inputs_count);
        params.inputs[0] = convert_data_tensor(comp_scale_past_layout);
        params.outputs[0] = convert_data_tensor(comp_scale_present_layout);

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;

        std::map<size_t, size_t> in_tensor_to_offset_map = {
            {0, in_offsets_map.at(3)}, // compression_scale_past
        };
        std::map<size_t, size_t> out_tensor_to_offset_map = {
            {0, out_offsets_map.at(2)}, // compression_scale_present
        };

        params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        return params;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<kv_cache>& arg, const kernel_impl_params& impl_param) {
        std::vector<kernel_selector::kernel_data> kernels_data;
        auto concat_kernel_params = get_concat_kernel_params(impl_param, impl_param.is_dynamic());
        auto& concat_kernel_selector = kernel_selector_t::Instance();
        kernels_data.push_back(concat_kernel_selector.get_best_kernel(concat_kernel_params));
        const bool indirect = impl_param.typed_desc<kv_cache>()->indirect;
        const bool compressed = impl_param.typed_desc<kv_cache>()->compressed;
        if (indirect) {
            auto bt_update_kernel_params = get_bt_update_kernel_params(impl_param, false);
            auto& bt_update_kernel_selector = bt_kernel_selector_t::Instance();
            kernels_data.push_back(bt_update_kernel_selector.get_best_kernel(bt_update_kernel_params));
        }

        if (compressed) {
            auto dq_kernel_params = get_dq_update_kernel_params(impl_param, impl_param.is_dynamic());
            auto& dq_kernel_selector = dq_kernel_selector_t::Instance();
            kernels_data.push_back(dq_kernel_selector.get_best_kernel(dq_kernel_params));

            auto concat_scale_zp_kernel_params = get_compression_scale_update_kernel_params(impl_param, impl_param.is_dynamic());
            auto& concat_scale_zp_kernel_selector = kernel_selector_t::Instance();
            kernels_data.push_back(concat_scale_zp_kernel_selector.get_best_kernel(concat_scale_zp_kernel_params));
        }
        return cldnn::make_unique<kv_cache_impl>(kernels_data);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernels_data[concat_stage].params == nullptr) {
            _kernels_data[concat_stage].params = std::make_shared<kernel_params_t>(get_concat_kernel_params(impl_param, true));
        }
        auto& params = static_cast<kernel_params_t&>(*_kernels_data[concat_stage].params);
        const auto inputs_count = 2;
        for (size_t i = 0; i < inputs_count; ++i) {
            params.inputs[i] = convert_data_tensor(impl_param.input_layouts[i]);
        }
        params.outputs[0] = convert_data_tensor(impl_param.output_layouts[0]);

        (_kernels_data[concat_stage].update_dispatch_data_func)(params, _kernels_data[concat_stage]);
        _kernels_data[concat_stage].kernels[0].skip_execution = impl_param._can_be_optimized || impl_param.get_input_layout(0).count() == 0;

        if (impl_param.typed_desc<kv_cache>()->compressed) {
            // In case of KV-cache with compression enabled, skip second concat's kernel as new token data append will
            // be handled by dynamic quantization kernel
            // However, allow execution of the first token for the case if KV-cache can't be optimized (if optimization is disabled, or
            // variables memory was reallocated and we have to copy past KV-cache to new memory)
            _kernels_data[concat_stage].kernels[1].skip_execution = true;

            // Update dynamic quantization parameters
            auto comp_scale_kernel_params = get_compression_scale_update_kernel_params(impl_param, impl_param.is_dynamic());
            (_kernels_data[scale_concat_stage].update_dispatch_data_func)(comp_scale_kernel_params, _kernels_data[scale_concat_stage]);
            _kernels_data[scale_concat_stage].kernels[0].skip_execution = impl_param._can_be_optimized || impl_param.get_input_layout(3).count() == 0;
        }
    }
};

namespace detail {

attach_kv_cache_impl::attach_kv_cache_impl() {
    auto types = { data_types::i8, data_types::f16, data_types::f32 };
    auto formats = { format::bfyx };
    implementation_map<kv_cache>::add(impl_types::ocl,
                                           shape_types::dynamic_shape,
                                           kv_cache_impl::create,
                                           types,
                                           formats);

    implementation_map<kv_cache>::add(impl_types::ocl,
                                           shape_types::static_shape,
                                           kv_cache_impl::create,
                                           types,
                                           formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::kv_cache_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::kv_cache)
