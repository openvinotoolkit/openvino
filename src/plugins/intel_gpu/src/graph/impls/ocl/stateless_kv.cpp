// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <limits>

#include "concatenation/concatenation_kernel_base.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "multi_stage_primitive.hpp"
#include "openvino/core/dimension.hpp"
#include "primitive_base.hpp"
#include "scatter_update/scatter_update_kernel_ref.h"
#include "scatter_update/scatter_update_kernel_selector.h"
#include "stateless_kv_inst.h"

namespace cldnn {
namespace ocl {

namespace {
enum class stateless_kv_stage : uint8_t {
    concat_to_sdpa,
    sdpa_to_present,
    new_to_present,
    new_scatter_present,
};

constexpr size_t swa_stage_count = 3;

kernel_selector::concat_axis convert_concat_axis(int64_t axis, size_t rank) {
    auto cldnn_axis = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    OPENVINO_ASSERT(cldnn_axis < static_cast<int64_t>(rank), "stateless_kv axis exceeds number of dimensions");

    // Difference in dimension ordering between OV and GPU plugin,
    // reverse spatial dimensions after batch and feature.
    if (cldnn_axis >= 2) {
        auto spatial_axis = cldnn_axis - 2;
        // Default and minimum number of dimensions is 4
        auto spatial_size = std::max<size_t>(rank, 4) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
    case 0:
        return kernel_selector::concat_axis::BATCH;
    case 1:
        return kernel_selector::concat_axis::FEATURE;
    case 2:
        return kernel_selector::concat_axis::X;
    case 3:
        return kernel_selector::concat_axis::Y;
    case 4:
        return kernel_selector::concat_axis::Z;
    case 5:
        return kernel_selector::concat_axis::W;
    default:
        OPENVINO_THROW("Unsupported stateless_kv axis: ", axis);
    }

    return kernel_selector::concat_axis::FEATURE;  // shouldn't get here
}

kernel_selector::scatter_update_axis convert_scatter_axis(int64_t axis, size_t rank) {
    auto cldnn_axis = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    OPENVINO_ASSERT(cldnn_axis < static_cast<int64_t>(rank), "stateless_kv axis exceeds number of dimensions");

    if (cldnn_axis >= 2) {
        auto spatial_axis = cldnn_axis - 2;
        auto spatial_size = std::max<size_t>(rank, 4) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
    case 0:
        return kernel_selector::scatter_update_axis::BATCH;
    case 1:
        return kernel_selector::scatter_update_axis::FEATURE;
    case 2:
        return kernel_selector::scatter_update_axis::X;
    case 3:
        return kernel_selector::scatter_update_axis::Y;
    case 4:
        return kernel_selector::scatter_update_axis::Z;
    case 5:
        return kernel_selector::scatter_update_axis::W;
    default:
        OPENVINO_THROW("Unsupported stateless_kv axis: ", axis);
    }

    return kernel_selector::scatter_update_axis::X;
}

}  // namespace

struct stateless_kv_impl : multi_stage_primitive<stateless_kv> {
    using parent = multi_stage_primitive<stateless_kv>;
    using scatter_kernel_selector_t = kernel_selector::scatter_update_kernel_selector;
    using scatter_kernel_params_t = kernel_selector::scatter_update_params;
    using concat_kernel_selector_t = kernel_selector::concatenation_kernel_selector;
    using concat_kernel_params_t = kernel_selector::concatenation_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::stateless_kv_impl)

    stateless_kv_impl() = default;

    stateless_kv_impl(const std::vector<kernel_selector::kernel_data>& kernels_data, const std::vector<stateless_kv_stage>& stages)
        : parent(kernels_data),
          _stages(stages) {
        OPENVINO_ASSERT(!kernels_data.empty());
        verify_stages();
        can_reuse_memory = kernels_data.front().can_reuse_memory;
    }

    stateless_kv_impl(const stateless_kv_impl& other)
        : parent(other),
          _stages(other._stages),
          _runtime_state(other._runtime_state),
          _is_inplace(other._is_inplace) {}

    std::unique_ptr<primitive_impl> clone() const override {
        auto cloned = std::make_unique<stateless_kv_impl>(*this);
        for (size_t i = 0; i < _stages.size(); ++i) {
            if (!_kernels_data[i].params)
                continue;

            if (_stages[i] == stateless_kv_stage::new_scatter_present) {
                const auto* params = dynamic_cast<const scatter_kernel_params_t*>(_kernels_data[i].params.get());
                OPENVINO_ASSERT(params != nullptr);
                cloned->_kernels_data[i].params = std::make_unique<scatter_kernel_params_t>(*params);
            } else {
                const auto* params = dynamic_cast<const concat_kernel_params_t*>(_kernels_data[i].params.get());
                OPENVINO_ASSERT(params != nullptr);
                cloned->_kernels_data[i].params = std::make_unique<concat_kernel_params_t>(*params);
            }
        }
        return cloned;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _stages.size();
        for (const auto stage : _stages)
            ob << static_cast<uint8_t>(stage);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        size_t stages_size = 0;
        ib >> stages_size;
        _stages.resize(stages_size);
        for (auto& stage : _stages) {
            uint8_t serialized_stage = 0;
            ib >> serialized_stage;
            stage = static_cast<stateless_kv_stage>(serialized_stage);
        }
        verify_stages();
        if ((!is_dynamic() && !is_swa()) || _kernels_data.front().kernelName.empty())
            return;

        for (size_t i = 0; i < _stages.size(); ++i) {
            auto& kernel_data = _kernels_data[i];
            switch (_stages[i]) {
            case stateless_kv_stage::concat_to_sdpa:
            case stateless_kv_stage::sdpa_to_present:
            case stateless_kv_stage::new_to_present: {
                auto& kernel_selector = concat_kernel_selector_t::Instance();
                auto kernel_impl = kernel_selector.GetImplementation(kernel_data.kernelName);
                kernel_impl->GetUpdateDispatchDataFunc(kernel_data);
                break;
            }
            case stateless_kv_stage::new_scatter_present: {
                auto& kernel_selector = scatter_kernel_selector_t::Instance();
                auto kernel_impl = kernel_selector.GetImplementation(kernel_data.kernelName);
                kernel_impl->GetUpdateDispatchDataFunc(kernel_data);
                break;
            }
            }
        }
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        auto& stateless_kv_instance = static_cast<stateless_kv_inst&>(inst);
        _runtime_state = stateless_kv_instance.get_runtime_state();
        _is_inplace = stateless_kv_instance.get_is_inplace();
        parent::update(inst, impl_params);
    }

    kernel_arguments_data get_arguments(const typed_primitive_inst<stateless_kv>& instance, size_t stage) const override {
        kernel_arguments_data args;
        auto past = instance.input_memory_ptr(0);
        if (!past) {
            past = instance.input_memory_ptr(1);
        }
        OPENVINO_ASSERT(stage < _stages.size());
        switch (_stages[stage]) {
        case stateless_kv_stage::concat_to_sdpa:
            args.inputs = {past, instance.input_memory_ptr(1)};
            args.outputs = {instance.output_memory_ptr(1)};
            break;
        case stateless_kv_stage::sdpa_to_present:
            args.inputs = {instance.output_memory_ptr(1)};
            args.outputs = {instance.output_memory_ptr(0)};
            break;
        case stateless_kv_stage::new_to_present:
            args.inputs = {past, instance.input_memory_ptr(1)};
            args.outputs = {instance.output_memory_ptr(0)};
            break;
        case stateless_kv_stage::new_scatter_present:
            args.inputs = {past, instance.input_memory_ptr(3), instance.input_memory_ptr(1)};
            args.outputs = {instance.output_memory_ptr(0)};
            break;
        }
        args.shape_info = instance.shape_info_memory_ptr();
        return args;
    }

    static std::optional<int64_t> get_concat_offset(const kernel_impl_params& impl_param) {
        const auto& desc = *impl_param.typed_desc<stateless_kv>();
        const auto& new_shape = impl_param.get_input_layout(1).get_partial_shape();
        const auto& present_shape = impl_param.get_output_layout(1).get_partial_shape();
        const auto& new_dim = new_shape[desc.concat_axis];
        const auto& present_dim = present_shape[desc.concat_axis];
        if (!present_dim.is_static())
            return {};
        OPENVINO_ASSERT(new_dim.is_static() && impl_param.get_input_layout(0).get_partial_shape()[desc.concat_axis].is_static());
        const auto begin = present_dim.get_length() - new_dim.get_length();
        OPENVINO_ASSERT(begin <= std::numeric_limits<uint32_t>::max(), "[GPU] stateless_kv update offset exceeds concat kernel scalar range");

        return begin;
    }

    static layout get_concat_input0_layout(const kernel_impl_params& impl_param, std::optional<int64_t> input0_len) {
        auto input0_layout = impl_param.get_input_layout(0);
        if (!input0_len)
            return input0_layout;

        const auto& primitive = impl_param.typed_desc<stateless_kv>();
        auto input0_shape = input0_layout.get_partial_shape();
        const auto input0_capacity = input0_shape[primitive->concat_axis].get_length();
        OPENVINO_ASSERT(*input0_len >= 0 && *input0_len <= input0_capacity);
        input0_shape[primitive->concat_axis] = *input0_len;
        input0_layout.set_partial_shape(input0_shape);
        input0_layout.data_padding._upper_size[primitive->concat_axis] += input0_capacity - *input0_len;
        return input0_layout;
    }

    static concat_kernel_params_t get_concat2_to1_kernel_params(const kernel_impl_params& impl_param,
                                                                const layout& input0_layout,
                                                                size_t input0_shape_info_idx,
                                                                size_t input1_idx,
                                                                size_t output_idx,
                                                                bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<stateless_kv>();
        auto params = get_default_params<concat_kernel_params_t>(impl_param, is_shape_agnostic);
        const auto& input1_layout = impl_param.get_input_layout(input1_idx);
        const auto& output_layout = impl_param.get_output_layout(output_idx);
        params.axis = convert_concat_axis(primitive->concat_axis, output_layout.get_rank());
        params.inputs.resize(2);
        params.inputs[0] = convert_data_tensor(input0_layout);
        params.inputs[1] = convert_data_tensor(input1_layout);
        params.outputs.resize(1);
        params.outputs[0] = convert_data_tensor(output_layout);
        params.kernelPerInput = true;

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;
        if (!in_offsets_map.empty() && !out_offsets_map.empty()) {
            std::map<size_t, size_t> in_tensor_to_offset_map = {
                {0, in_offsets_map.at(input0_shape_info_idx)},
                {1, in_offsets_map.at(input1_idx)},
            };
            std::map<size_t, size_t> out_tensor_to_offset_map = {
                {0, out_offsets_map.at(output_idx)},
            };
            params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);
        }
        return params;
    }

    static concat_kernel_params_t get_swa_concat_kernel_params(const kernel_impl_params& impl_param,
                                                               size_t stage,
                                                               const layout& input_layout,
                                                               const layout& output_layout,
                                                               size_t input_shape_info_idx,
                                                               size_t output_shape_info_idx,
                                                               bool is_shape_agnostic) {
        const auto& primitive = impl_param.typed_desc<stateless_kv>();
        auto params = get_default_params<concat_kernel_params_t>(impl_param, is_shape_agnostic);
        const auto concat_axis = primitive->concat_axis;
        OPENVINO_ASSERT(concat_axis >= 0 && concat_axis < static_cast<int64_t>(output_layout.get_rank()));
        auto stage_input_layout = input_layout;
        if (is_shape_agnostic) {
            auto input_shape = stage_input_layout.get_partial_shape();
            input_shape[concat_axis] = ov::Dimension::dynamic();
            stage_input_layout.set_partial_shape(input_shape);
            stage_input_layout.data_padding._dynamic_dims_mask[concat_axis] = 1;
        }
        params.axis = convert_concat_axis(primitive->concat_axis, output_layout.get_rank());
        params.inputs.resize(1);
        params.inputs[0] = convert_data_tensor(stage_input_layout);
        params.outputs.resize(1);
        params.outputs[0] = convert_data_tensor(output_layout);
        params.kernelPerInput = true;
        params.is_shape_agnostic = params.has_dynamic_tensors();
        params.stage_id = stage;
        params.set_dynamic_shape_offsets({{0, impl_param.in_port_to_shape_info_offset.at(input_shape_info_idx)}},
                                         {{0, impl_param.out_port_to_shape_info_offset.at(output_shape_info_idx)}});
        return params;
    }

    static scatter_kernel_params_t get_scatter_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<stateless_kv>();
        GPU_DEBUG_TRACE_DETAIL << primitive->id << ": get_kernel_params in[" << impl_param.get_input_layout(0).to_short_string() << "] out["
                               << impl_param.get_output_layout(0).to_short_string() << "][" << impl_param.get_output_layout(1).to_short_string() << "]"
                               << std::endl;
        auto params = get_default_params<kernel_selector::scatter_update_params>(impl_param, is_shape_agnostic);

        params.axis = convert_scatter_axis(primitive->concat_axis, impl_param.get_input_layout(0).get_rank());
        params.inputs.resize(3);
        params.inputs[0] = convert_data_tensor(impl_param.get_input_layout(0));
        params.inputs[1] = convert_data_tensor(impl_param.get_input_layout(3));
        params.inputs[2] = convert_data_tensor(impl_param.get_input_layout(1));
        params.outputs.resize(1);
        params.outputs[0] = convert_data_tensor(impl_param.get_output_layout(0));
        params.is_inplace = false;

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;

        if (!in_offsets_map.empty() && !out_offsets_map.empty()) {
            std::map<size_t, size_t> in_tensor_to_offset_map = {
                {0, in_offsets_map.at(0)},
                {1, in_offsets_map.at(3)},
                {2, in_offsets_map.at(1)},
            };
            std::map<size_t, size_t> out_tensor_to_offset_map = {
                {0, out_offsets_map.at(0)},
            };
            params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);
        }
        return params;
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<stateless_kv>& arg, const kernel_impl_params& impl_param) {
        auto params = static_canonicalize_shapes(impl_param);
        const auto& primitive = params.typed_desc<stateless_kv>();
        const auto swa = primitive->window_size > 0;
        std::vector<kernel_selector::kernel_data> kernels_data;
        std::vector<stateless_kv_stage> stages;
        kernels_data.reserve(swa ? swa_stage_count : 1);
        stages.reserve(swa ? swa_stage_count : 1);

        if (primitive->window_size > 0) {
            OPENVINO_ASSERT(primitive->input.size() == 3, "[GPU] stateless_kv doesn't support window_size together with pos_idx");
            auto& kernel_selector = concat_kernel_selector_t::Instance();
            auto past_view_layout = params.get_input_layout(0);
            auto past_view_shape = past_view_layout.get_partial_shape();
            past_view_shape[primitive->concat_axis] = ov::Dimension::dynamic();
            past_view_layout.set_partial_shape(past_view_shape);
            past_view_layout.data_padding._dynamic_dims_mask[primitive->concat_axis] = 1;
            auto concat_to_sdpa_params = get_concat2_to1_kernel_params(params, past_view_layout, 3, 1, 1, true);
            concat_to_sdpa_params.stage_id = stages.size();
            auto concat_to_sdpa_data = kernel_selector.get_best_kernel(concat_to_sdpa_params);
            OPENVINO_ASSERT(concat_to_sdpa_data.kernels.size() == 2, "[GPU] stateless_kv concat-to-SDPA stage expects two sub-kernels");
            kernels_data.push_back(std::move(concat_to_sdpa_data));
            stages.push_back(stateless_kv_stage::concat_to_sdpa);

            auto sdpa_to_present_params =
                get_swa_concat_kernel_params(params, stages.size(), params.get_output_layout(1), params.get_output_layout(0), 4, 0, true);
            auto sdpa_to_present_data = kernel_selector.get_best_kernel(sdpa_to_present_params);
            OPENVINO_ASSERT(sdpa_to_present_data.kernels.size() == 1, "[GPU] stateless_kv SDPA-to-present stage expects one sub-kernel");
            kernels_data.push_back(std::move(sdpa_to_present_data));
            stages.push_back(stateless_kv_stage::sdpa_to_present);
        }

        if (primitive->input.size() == 3) {
            auto input0_layout = params.get_input_layout(0);
            auto input0_shape = input0_layout.get_partial_shape();
            input0_shape[primitive->concat_axis] = ov::Dimension::dynamic();
            input0_layout.set_partial_shape(input0_shape);
            input0_layout.data_padding._dynamic_dims_mask[primitive->concat_axis] = 1;
            auto kernel_params = get_concat2_to1_kernel_params(params, input0_layout, 0, 1, 0, true);
            kernel_params.stage_id = stages.size();
            kernel_params.is_shape_agnostic = kernel_params.has_dynamic_tensors();
            auto& kernel_selector = concat_kernel_selector_t::Instance();
            auto best_kernel = kernel_selector.get_best_kernel(kernel_params);
            OPENVINO_ASSERT(best_kernel.kernels.size() == 2, "[GPU] stateless_kv concat expects two sub-kernels");
            kernels_data.push_back(std::move(best_kernel));
            stages.push_back(stateless_kv_stage::new_to_present);
        } else {
            OPENVINO_ASSERT(!swa, "[GPU] stateless_kv SWA doesn't support scatter update");
            auto kernel_params = get_scatter_kernel_params(params, impl_param.is_dynamic());
            kernel_params.is_shape_agnostic = impl_param.is_dynamic();
            auto& kernel_selector = scatter_kernel_selector_t::Instance();
            auto best_kernel = kernel_selector.get_best_kernel(kernel_params);
            kernels_data.push_back(std::move(best_kernel));
            stages.push_back(stateless_kv_stage::new_scatter_present);
        }
        return std::make_unique<stateless_kv_impl>(kernels_data, stages);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        const auto swa = is_swa();
        const auto* state = swa ? &get_swa_runtime_state() : nullptr;
        const auto rolling = state != nullptr && state->is_rolling();
        if (state != nullptr) {
            OPENVINO_ASSERT(!state->is_rolling() || (state->past_front_idx <= state->sdpa_front_idx && state->sdpa_front_idx <= state->present_front_idx));
            OPENVINO_ASSERT(state->get_new_in_sdpa_offset() >= 0);
            OPENVINO_ASSERT(state->get_present_output_len() == std::max<int64_t>(0, state->get_new_in_present_offset()) + state->new_token_len -
                                                                   std::max<int64_t>(0, -state->get_new_in_present_offset()));
        }

        const auto& primitive = impl_param.typed_desc<stateless_kv>();
        for (size_t stage_idx = 0; stage_idx < _stages.size(); ++stage_idx) {
            const auto stage = _stages[stage_idx];
            auto& kernel_data = _kernels_data[stage_idx];
            const auto skip_stage = [&kernel_data]() {
                for (auto& kernel : kernel_data.kernels)
                    kernel.skip_execution = true;
            };

            switch (stage) {
            case stateless_kv_stage::concat_to_sdpa: {
                OPENVINO_ASSERT(swa && state != nullptr, "[GPU] stateless_kv concat-to-SDPA stage requires SWA");
                if (!rolling || (state->get_new_in_sdpa_offset() == 0 && state->new_token_len == 0)) {
                    skip_stage();
                    break;
                }
                const auto past_view = state->get_past_to_sdpa_input_layout(impl_param.get_input_layout(0), primitive->concat_axis);
                auto updated_params = get_concat2_to1_kernel_params(impl_param, past_view, 3, 1, 1, true);
                updated_params.stage_id = stage_idx;
                if (!kernel_data.params) {
                    kernel_data.params = std::make_shared<concat_kernel_params_t>(std::move(updated_params));
                } else {
                    static_cast<concat_kernel_params_t&>(*kernel_data.params) = std::move(updated_params);
                }
                kernel_data.update_dispatch_data_func(*kernel_data.params, kernel_data);
                OPENVINO_ASSERT(kernel_data.kernels.size() == 2);
                for (auto& kernel : kernel_data.kernels)
                    kernel.skip_execution = false;
                break;
            }
            case stateless_kv_stage::sdpa_to_present: {
                OPENVINO_ASSERT(swa && state != nullptr, "[GPU] stateless_kv SDPA-to-present stage requires SWA");
                if (!rolling || state->get_present_output_len() == 0) {
                    skip_stage();
                    break;
                }
                const auto sdpa_view = state->get_sdpa_to_present_input_layout(impl_param.get_output_layout(1), primitive->concat_axis);
                auto updated_params = get_swa_concat_kernel_params(impl_param, stage_idx, sdpa_view, impl_param.get_output_layout(0), 4, 0, false);
                if (!kernel_data.params) {
                    kernel_data.params = std::make_shared<concat_kernel_params_t>(std::move(updated_params));
                } else {
                    static_cast<concat_kernel_params_t&>(*kernel_data.params) = std::move(updated_params);
                }
                kernel_data.update_dispatch_data_func(*kernel_data.params, kernel_data);
                OPENVINO_ASSERT(kernel_data.kernels.size() == 1);
                kernel_data.kernels.front().skip_execution = false;
                auto& scalars = kernel_data.kernels.front().params.scalars;
                OPENVINO_ASSERT(scalars.size() == 1, "[GPU] stateless_kv SWA concat stage expects one destination offset scalar");
                scalars[0].v.u32 = 0;
                break;
            }
            case stateless_kv_stage::new_to_present: {
                if (swa && rolling) {
                    skip_stage();
                    break;
                }
                const auto concat_offset = swa ? std::optional<int64_t>(state->get_new_in_present_offset()) : get_concat_offset(impl_param);
                const auto input0_layout = get_concat_input0_layout(impl_param, concat_offset);
                auto updated_params = get_concat2_to1_kernel_params(impl_param, input0_layout, 0, 1, 0, true);
                updated_params.stage_id = stage_idx;
                if (!kernel_data.params) {
                    kernel_data.params = std::make_shared<concat_kernel_params_t>(std::move(updated_params));
                } else {
                    static_cast<concat_kernel_params_t&>(*kernel_data.params) = std::move(updated_params);
                }
                kernel_data.update_dispatch_data_func(*kernel_data.params, kernel_data);
                OPENVINO_ASSERT(kernel_data.kernels.size() == 2, "[GPU] stateless_kv concat expects two sub-kernels");
                kernel_data.kernels[0].skip_execution = _is_inplace;
                kernel_data.kernels[1].skip_execution = false;
                if (concat_offset) {
                    OPENVINO_ASSERT(*concat_offset >= 0 && *concat_offset <= std::numeric_limits<uint32_t>::max());
                    auto& scalars = kernel_data.kernels[1].params.scalars;
                    OPENVINO_ASSERT(scalars.size() == 1, "[GPU] stateless_kv concat append kernel expects one offset scalar");
                    scalars[0].v.u32 = static_cast<uint32_t>(*concat_offset);
                }
                break;
            }
            case stateless_kv_stage::new_scatter_present: {
                OPENVINO_ASSERT(!swa, "[GPU] stateless_kv scatter stage doesn't support SWA");
                auto updated_params = get_scatter_kernel_params(impl_param, true);
                updated_params.is_inplace = _is_inplace;
                if (!kernel_data.params) {
                    kernel_data.params = std::make_shared<scatter_kernel_params_t>(std::move(updated_params));
                } else {
                    static_cast<scatter_kernel_params_t&>(*kernel_data.params) = std::move(updated_params);
                }
                kernel_data.update_dispatch_data_func(*kernel_data.params, kernel_data);
                break;
            }
            }
        }
    }

    void set_arguments_impl(stateless_kv_inst&) override {}

private:
    const stateless_kv_runtime_state& get_swa_runtime_state() const {
        OPENVINO_ASSERT(_runtime_state.has_value(), "[GPU] stateless_kv SWA runtime state wasn't initialized during shape update");
        return *_runtime_state;
    }

    event::ptr execute_stage(const std::vector<event::ptr>& events, stateless_kv_inst& instance, size_t stage) {
        OPENVINO_ASSERT(stage < _kernels_data.size());
        auto& stream = instance.get_network().get_stream();
        auto tmp_events = events;
        std::vector<event::ptr> all_events;
        size_t kernel_offset = 0;
        for (size_t current_stage = 0; current_stage < stage; ++current_stage) {
            kernel_offset += _kernels_data[current_stage].kernels.size();
        }

        auto& kernel_data = _kernels_data[stage];
        for (size_t kernel_idx = 0; kernel_idx < kernel_data.kernels.size(); ++kernel_idx) {
            if (kernel_data.kernels[kernel_idx].skip_execution)
                continue;

            const auto compiled_kernel_idx = kernel_offset + kernel_idx;
            auto& params = kernel_data.kernels[kernel_idx].params;
            auto args = get_arguments(instance, stage);
            args.scalars = &params.scalars;
            args.local_memory_args = &params.local_memory_args;
            for (const auto& memory : instance.get_intermediates_memories()) {
                args.intermediates.push_back(memory);
            }

            stream.set_arguments(*_kernels[compiled_kernel_idx], params, args);
            auto event = stream.enqueue_kernel(*_kernels[compiled_kernel_idx], params, args, tmp_events, instance.needs_completion_event() || is_swa());
            if (kernel_data.needs_sub_kernels_sync)
                tmp_events = {event};
            all_events.push_back(event);
            kernel_dump_info.add_entry_point(_kernels[compiled_kernel_idx]->get_id());
        }

        if (all_events.empty())
            return stream.aggregate_events(tmp_events, tmp_events.size() > 1, instance.is_output());
        return stream.aggregate_events(all_events, all_events.size() > 1);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, stateless_kv_inst& instance) override {
        kernel_dump_info.clear_entries();
        if (instance.can_be_optimized()) {
            auto& stream = instance.get_network().get_stream();
            return stream.aggregate_events(events, events.size() > 1, instance.is_output());
        }
        OPENVINO_ASSERT(!_stages.empty());
        event::ptr last_event;
        for (size_t stage_idx = 0; stage_idx < _stages.size(); ++stage_idx) {
            const auto& dependencies = last_event ? std::vector<event::ptr>{last_event} : events;
            last_event = execute_stage(dependencies, instance, stage_idx);
        }
        return last_event;
    }

    bool is_swa() const noexcept {
        return _stages.size() == swa_stage_count;
    }

    void verify_stages() const {
        OPENVINO_ASSERT(_kernels_data.size() == _stages.size(), "[GPU] stateless_kv kernel data and stage counts don't match");
        if (is_swa()) {
            OPENVINO_ASSERT(_stages[0] == stateless_kv_stage::concat_to_sdpa && _stages[1] == stateless_kv_stage::sdpa_to_present &&
                                _stages[2] == stateless_kv_stage::new_to_present,
                            "[GPU] stateless_kv has invalid SWA stages");
        } else {
            OPENVINO_ASSERT(
                _stages.size() == 1 && (_stages.front() == stateless_kv_stage::new_to_present || _stages.front() == stateless_kv_stage::new_scatter_present),
                "[GPU] stateless_kv has invalid non-SWA stage");
        }
    }

    std::vector<stateless_kv_stage> _stages;
    std::optional<stateless_kv_runtime_state> _runtime_state;
    bool _is_inplace = false;
};

namespace detail {

attach_stateless_kv_impl::attach_stateless_kv_impl() {
    // u8 / f8e4m3 cover compressed KV caches (u8 also carries packed i4); concat_ref and scatter_update_ref support them.
    auto types = {data_types::i8, data_types::u8, data_types::f8e4m3, data_types::f16, data_types::f32};
    auto formats = {format::bfyx};
    implementation_map<stateless_kv>::add(impl_types::ocl, shape_types::dynamic_shape, stateless_kv_impl::create, types, formats);
    implementation_map<stateless_kv>::add(impl_types::ocl, shape_types::static_shape, stateless_kv_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::stateless_kv_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::stateless_kv)
