// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include "primitive_base.hpp"

#include "kv_cache_inst.h"
#include "concatenation/concatenation_kernel_selector.h"
#include "concatenation/concatenation_kernel_base.h"

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::concat_axis convert_axis(int64_t axis, size_t rank) {
    auto cldnn_axis = axis >= 0 ? axis : axis + static_cast<int64_t>(rank);
    if (cldnn_axis >= static_cast<int64_t>(rank))
        OPENVINO_THROW("kv_cache axis exceeds number of dimensions");

    // Difference in dimension ordering between IE and GPU plugin,
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

struct kv_cache_impl : typed_primitive_impl_ocl<kv_cache> {
    using parent = typed_primitive_impl_ocl<kv_cache>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::concatenation_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::concatenation_params, kernel_selector::concatenation_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::kv_cache_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<kv_cache_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    kernel_arguments_data get_arguments(const kv_cache_inst& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        args.inputs = { instance.input_memory_ptr(0), instance.input_memory_ptr(1) };

        return args;
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, kv_cache_inst& instance) override {
        const bool can_be_optimized = instance.get_impl_params()->_can_be_optimized;
        const auto& desc = instance.get_typed_desc<kv_cache>();
        auto& variable = instance.get_network().get_variable(desc->variable_info.variable_id);
        auto res_event = parent::execute_impl(events, instance);
        variable.set();

        if (can_be_optimized) {
            GPU_DEBUG_TRACE_DETAIL << desc->id  << " : Output is same as variable memory! Skip copying " << std::endl;
            // When primitive is optimized, concat kernel writes directly to variable memory
            return res_event;
        } else {
            // Othwerise, we need to copy result from out buffer to state memory
            GPU_DEBUG_TRACE_DETAIL << desc->id  << " : Copying output to variable meomry" << std::endl;
            auto& stream = instance.get_network().get_stream();

            stream.enqueue_barrier();
            auto out = instance.get_network().get_engine().reinterpret_buffer(instance.output_memory(0), variable.get_memory()->get_layout());
            return variable.get_memory()->copy_from(stream, *out, false);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<kv_cache>();
        auto params = get_default_params<kernel_selector::concatenation_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::concatenation_optional_params>(impl_param.get_program());
        auto axis = primitive->concat_axis;

        const auto inputs_count = primitive->input_size();
        params.inputs.resize(inputs_count);
        for (size_t i = 0; i < inputs_count; ++i) {
            params.inputs[i] = convert_data_tensor(impl_param.input_layouts[i]);
        }

        params.axis = convert_axis(axis, impl_param.get_output_layout().get_rank());
        optional_params.kernelPerInput = true;

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
        _kernel_data.kernels[0].skip_execution = impl_param._can_be_optimized || impl_param.get_input_layout(0).count() == 0;
    }
};

namespace detail {

attach_kv_cache_impl::attach_kv_cache_impl() {
    auto types = { data_types::f16, data_types::f32 };
    auto formats = { format::bfyx };
    implementation_map<kv_cache>::add(impl_types::ocl,
                                           shape_types::dynamic_shape,
                                           typed_primitive_impl_ocl<kv_cache>::create<kv_cache_impl>,
                                           types,
                                           formats);

    implementation_map<kv_cache>::add(impl_types::ocl,
                                           shape_types::static_shape,
                                           typed_primitive_impl_ocl<kv_cache>::create<kv_cache_impl>,
                                           types,
                                           formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::kv_cache_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::kv_cache)
