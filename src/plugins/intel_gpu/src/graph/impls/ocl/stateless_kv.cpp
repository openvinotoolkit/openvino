// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "primitive_base.hpp"

#include "stateless_kv_inst.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "concatenation/concatenation_kernel_base.h"
#include "scatter_update/scatter_update_kernel_selector.h"
#include "scatter_update/scatter_update_kernel_ref.h"
#include "openvino/core/dimension.hpp"

#include <cstdint>
#include <limits>

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::concat_axis convert_concat_axis(int64_t axis, size_t rank) {
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

kernel_selector::scatter_update_axis convert_scatter_axis(int64_t axis, size_t rank) {
    auto cldnn_axis = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    if (cldnn_axis >= static_cast<int64_t>(rank))
        OPENVINO_THROW("stateless_kv axis exceeds number of dimensions");

    if (cldnn_axis >= 2) {
        auto spatial_axis = cldnn_axis - 2;
        auto spatial_size = std::max<size_t>(rank, 4) - 2;
        cldnn_axis = spatial_size - spatial_axis - 1 + 2;
    }

    switch (cldnn_axis) {
        case 0: return kernel_selector::scatter_update_axis::BATCH;
        case 1: return kernel_selector::scatter_update_axis::FEATURE;
        case 2: return kernel_selector::scatter_update_axis::X;
        case 3: return kernel_selector::scatter_update_axis::Y;
        case 4: return kernel_selector::scatter_update_axis::Z;
        case 5: return kernel_selector::scatter_update_axis::W;
        default: OPENVINO_THROW("Unsupported stateless_kv axis: ", axis);
    }

    return kernel_selector::scatter_update_axis::X;
}

}  // namespace

struct stateless_kv_impl : typed_primitive_impl_ocl<stateless_kv> {
    using parent = typed_primitive_impl_ocl<stateless_kv>;
    using parent::parent;
    using scatter_kernel_selector_t = kernel_selector::scatter_update_kernel_selector;
    using scatter_kernel_params_t = kernel_selector::scatter_update_params;
    using concat_kernel_selector_t = kernel_selector::concatenation_kernel_selector;
    using concat_kernel_params_t = kernel_selector::concatenation_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::stateless_kv_impl)

    stateless_kv_impl() = default;

    stateless_kv_impl(const kernel_selector::kernel_data& kd, bool has_pos_idx_input)
        : parent(kd), _has_pos_idx_input(has_pos_idx_input) {}

    std::unique_ptr<primitive_impl> clone() const override {
        if (!_has_pos_idx_input)
            return make_deep_copy<stateless_kv_impl, concat_kernel_params_t>(*this);
        return make_deep_copy<stateless_kv_impl, scatter_kernel_params_t>(*this);
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
        ob << _has_pos_idx_input;
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        ib >> _has_pos_idx_input;
        if (!is_dynamic() || _kernel_data.kernelName.empty())
            return;

        if (_has_pos_idx_input) {
            auto& kernel_selector = scatter_kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        } else {
            auto& kernel_selector = concat_kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    kernel_arguments_data get_arguments(const typed_primitive_inst<stateless_kv>& instance) const override {
        kernel_arguments_data args;
        args.inputs.push_back(instance.input_memory_ptr(0)); // past
        if (!_has_pos_idx_input) {
            args.inputs.push_back(instance.input_memory_ptr(1)); // new_token_data
        } else {
            args.inputs.push_back(instance.input_memory_ptr(3)); // pos_idx
            args.inputs.push_back(instance.input_memory_ptr(1)); // new_token_data
        }
        args.outputs.push_back(instance.output_memory_ptr(0));
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
        OPENVINO_ASSERT(begin <= std::numeric_limits<uint32_t>::max(),
                        "[GPU] stateless_kv update offset exceeds concat kernel scalar range");

        return begin;
    }

    static concat_kernel_params_t get_concat_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<stateless_kv>();
        auto params = get_default_params<concat_kernel_params_t>(impl_param, is_shape_agnostic);
        auto past_layout = impl_param.get_input_layout(0);
        const auto concat_offset = get_concat_offset(impl_param);
        if (concat_offset) {
            auto copy_shape = past_layout.get_partial_shape();
            const auto past_length = copy_shape[primitive->concat_axis].get_length();
            copy_shape[primitive->concat_axis] = *concat_offset;
            past_layout.set_partial_shape(copy_shape);
            past_layout.data_padding._upper_size[primitive->concat_axis] += past_length - *concat_offset;
        }
        params.axis = convert_concat_axis(primitive->concat_axis, impl_param.get_output_layout(0).get_rank());
        params.inputs.resize(2);
        params.inputs[0] = convert_data_tensor(past_layout);
        params.inputs[1] = convert_data_tensor(impl_param.get_input_layout(1));
        params.outputs.resize(1);
        params.outputs[0] = convert_data_tensor(impl_param.get_output_layout(0));
        params.kernelPerInput = true;

        const auto& in_offsets_map = impl_param.in_port_to_shape_info_offset;
        const auto& out_offsets_map = impl_param.out_port_to_shape_info_offset;
        if (!in_offsets_map.empty() && !out_offsets_map.empty()) {
            std::map<size_t, size_t> in_tensor_to_offset_map = {
                {0, in_offsets_map.at(0)},
                {1, in_offsets_map.at(1)},
            };
            std::map<size_t, size_t> out_tensor_to_offset_map = {
                {0, out_offsets_map.at(0)},
            };
            params.set_dynamic_shape_offsets(in_tensor_to_offset_map, out_tensor_to_offset_map);

        }
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
        if (params.typed_desc<stateless_kv>()->input.size() == 3) {
            auto kernel_params = get_concat_kernel_params(params, impl_param.is_dynamic());
            kernel_params.is_shape_agnostic = kernel_params.has_dynamic_tensors();
            auto& kernel_selector = concat_kernel_selector_t::Instance();
            auto best_kernel = kernel_selector.get_best_kernel(kernel_params);
            OPENVINO_ASSERT(best_kernel.kernels.size() == 2, "[GPU] stateless_kv concat expects two sub-kernels");
            return std::make_unique<stateless_kv_impl>(best_kernel, false);
        } else {
            auto kernel_params = get_scatter_kernel_params(params, impl_param.is_dynamic());
            kernel_params.is_shape_agnostic = impl_param.is_dynamic();
            auto& kernel_selector = scatter_kernel_selector_t::Instance();
            auto best_kernel = kernel_selector.get_best_kernel(kernel_params);
            return std::make_unique<stateless_kv_impl>(best_kernel, true);
        }
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        if (!_has_pos_idx_input) {
            if (_kernel_data.params == nullptr) {
                _kernel_data.params = std::make_shared<concat_kernel_params_t>(get_concat_kernel_params(impl_param, true));
            } else {
                static_cast<concat_kernel_params_t&>(*_kernel_data.params) = get_concat_kernel_params(impl_param, true);
            }
            (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
            const auto concat_offset = get_concat_offset(impl_param);
            if (concat_offset) {
                auto& scalars = _kernel_data.kernels[1].params.scalars;
                OPENVINO_ASSERT(scalars.size() == 1, "[GPU] stateless_kv concat append kernel expects one offset scalar");
                scalars[0].v.u32 = static_cast<uint32_t>(*concat_offset);
            }
        } else {
            if (_kernel_data.params == nullptr) {
                _kernel_data.params = std::make_shared<scatter_kernel_params_t>(get_scatter_kernel_params(impl_param, true));
            } else {
                static_cast<scatter_kernel_params_t&>(*_kernel_data.params) = get_scatter_kernel_params(impl_param, true);
            }
            (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
        }
    }

    void set_arguments_impl(stateless_kv_inst& instance) override {
        if (!_has_pos_idx_input) {
            update_dispatch_data(*instance.get_impl_params());
            OPENVINO_ASSERT(_kernel_data.kernels.size() == 2, "[GPU] stateless_kv concat expects two sub-kernels");
            _kernel_data.kernels[0].skip_execution = instance.get_is_inplace() || _kernel_data.kernels[0].skip_execution;
        } else {
            if (!_kernel_data.params) {
                update_dispatch_data(*instance.get_impl_params());
            }
            auto& params = static_cast<scatter_kernel_params_t&>(*_kernel_data.params);
            if (params.is_inplace != instance.get_is_inplace()) {
                params.is_inplace = instance.get_is_inplace();
                (_kernel_data.update_dispatch_data_func)(params, _kernel_data);
            }
        }
        parent::set_arguments_impl(instance);
    }

private:
    bool _has_pos_idx_input = false;
};

namespace detail {

attach_stateless_kv_impl::attach_stateless_kv_impl() {
    auto types = {data_types::i8, data_types::f16, data_types::f32};
    auto formats = {format::bfyx};
    implementation_map<stateless_kv>::add(impl_types::ocl, shape_types::dynamic_shape, stateless_kv_impl::create, types, formats);
    implementation_map<stateless_kv>::add(impl_types::ocl, shape_types::static_shape, stateless_kv_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::stateless_kv_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::stateless_kv)
