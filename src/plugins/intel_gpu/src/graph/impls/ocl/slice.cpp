// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "slice_inst.h"
#include "data_inst.h"
#include "slice/slice_kernel_selector.h"
#include "slice/slice_kernel_ref.h"

#include <algorithm>
#include <cstddef>

namespace cldnn {
namespace ocl {

namespace {
template<typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
std::vector<std::int32_t> extractIntegerData(const data_node& node, const stream& stream) {
    mem_lock<T> lock{node.get_attached_memory_ptr(), stream};
    T* data = lock.data();
    std::vector<std::int32_t> integer_data;
    integer_data.reserve(node.get_output_layout().count());
    for (size_t i = 0; i < node.get_output_layout().count(); i++) {
        integer_data.emplace_back(static_cast<std::int32_t>(data[i]));
    }
    return integer_data;
}

std::vector<std::int32_t> extractIntegerData(const data_node& node, const stream& stream) {
    auto dt = node.get_output_layout().data_type;
    switch (dt) {
    case data_types::u8:
        return extractIntegerData<std::uint8_t>(node, stream);
    case data_types::i8:
        return extractIntegerData<std::int8_t>(node, stream);
    case data_types::i32:
        return extractIntegerData<std::int32_t>(node, stream);
    case data_types::i64:
        return extractIntegerData<std::int64_t>(node, stream);
    default:
        OPENVINO_ASSERT(false, "[GPU] Slice parameters should be of integral type for node ", node.id(), " while got ", dt);
    }
    return {};
}

} // namespace

struct slice_impl : typed_primitive_impl_ocl<slice> {
    using parent = typed_primitive_impl_ocl<slice>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::slice_kernel_selector;
    using kernel_params_t = kernel_selector::slice_params;

    enum InputIndices {
        kData,
        kStart,
        kEnd,
        kStep,
        kAxes,
        kInputsNum
    };

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::slice_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<slice_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    kernel_arguments_data get_arguments(const slice_inst& instance) const override {
        kernel_arguments_data args;

        const slice_node* node = instance.node;

        const auto& inputs = node->get_dependencies();

        const bool axes_in_runtime_time = ((inputs.size() == InputIndices::kInputsNum) && !inputs[InputIndices::kAxes].first->is_constant());
        const bool start_in_runtime = axes_in_runtime_time || inputs[InputIndices::kStart].first->is_dynamic();
        const bool step_in_runtime = axes_in_runtime_time || inputs[InputIndices::kStep].first->is_dynamic();

        args.inputs.push_back(instance.input_memory_ptr(0));

        if (start_in_runtime)
            args.inputs.push_back(instance.input_memory_ptr(1));

        if (step_in_runtime)
            args.inputs.push_back(instance.input_memory_ptr(3));

        if (axes_in_runtime_time)
            args.inputs.push_back(instance.input_memory_ptr(4));

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();
        return args;
    }

    static std::unique_ptr<primitive_impl> create(const slice_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::slice_params>(impl_param, impl_param.is_dynamic());
        const auto& inputs = arg.get_dependencies();
        const stream& stream = arg.get_program().get_stream();
        const auto input_rank = params.inputs[0].Dimentions();

        if (inputs.size() == InputIndices::kInputsNum) {
            params.axes_data_type = inputs[InputIndices::kAxes].first->get_output_layout(0).data_type;

            // Prepare constant time axis if avaiable.
            if (inputs[InputIndices::kAxes].first->is_constant()) {
                params.compile_time_axes = extractIntegerData(inputs[InputIndices::kAxes].first->as<data>(), stream);
                for (size_t axis = 0; axis < params.compile_time_axes.size(); axis++) {
                    const auto transformed_axe = params.compile_time_axes[axis] < 0
                                                     ? input_rank + params.compile_time_axes[axis]
                                                     : params.compile_time_axes[axis];
                    params.compile_time_axes[axis] = transformed_axe;
                }
            } else {
                params.compile_time_axes.clear();
                auto layout = impl_param.get_input_layout(InputIndices::kAxes);
                params.inputs.push_back(convert_data_tensor(layout));
            }
        } else {
            params.axes_data_type = ov::element::i32;
            params.compile_time_axes.resize(input_rank);
            std::iota(params.compile_time_axes.begin(), params.compile_time_axes.end(), 0);
        }

        const bool axes_in_compile_time = !params.compile_time_axes.empty();

        params.start_data_type = inputs[InputIndices::kStart].first->get_output_layout(0).data_type;
        if (inputs[InputIndices::kStart].first->is_constant() && axes_in_compile_time) {
            params.compile_time_start = extractIntegerData(inputs[InputIndices::kStart].first->as<data>(), stream);
        } else {
            params.compile_time_start.clear();
            auto layout = impl_param.get_input_layout(InputIndices::kStart);
            params.inputs.push_back(convert_data_tensor(layout));
        }

        params.step_data_type = inputs[InputIndices::kStep].first->get_output_layout(0).data_type;
        if (inputs[InputIndices::kStep].first->is_constant() && axes_in_compile_time) {
            params.compile_time_step = extractIntegerData(inputs[InputIndices::kStep].first->as<data>(), stream);
        } else {
            params.compile_time_step.clear();
            auto stop_layout = impl_param.get_input_layout(InputIndices::kStep);
            params.inputs.push_back(convert_data_tensor(stop_layout));
        }

        // NOTE: Stop input is not used by the slice kernel, as this information
        // is implicitely passed with output shape.

        params.set_dynamic_shape_offsets();
        auto &kernel_selector =
                kernel_selector::slice_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(params);

        return make_unique<slice_impl>(best_kernel);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_default_params<kernel_selector::slice_params>(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params, _kernel_data);
    }
};

namespace detail {

attach_slice_impl::attach_slice_impl() {
    auto types = {data_types::f32, data_types::f16, data_types::i8, data_types::u8, data_types::i32, data_types::i64};

    auto formats = {
        format::bfyx,
        format::bfzyx,
    };

    implementation_map<slice>::add(impl_types::ocl, shape_types::any, slice_impl::create, types, formats);
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::slice_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::slice)
