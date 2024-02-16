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

std::vector<std::int32_t> extractShape(kernel_selector::Tensor::DataTensor& tensor) {
    auto logical_dims = tensor.LogicalDims();
    // LogicalDims method returns dims in reversed order
    std::vector<int32_t> reverse_logical_dims;
    for (auto it = logical_dims.rbegin(); it != logical_dims.rend(); ++it) {
        reverse_logical_dims.push_back(static_cast<int32_t>(*it));
    }
    return reverse_logical_dims;
}

} // namespace

struct slice_impl : typed_primitive_impl_ocl<slice> {
    using parent = typed_primitive_impl_ocl<slice>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::slice_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::slice_params, kernel_selector::slice_optional_params>;

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
        kernel_selector::slice_params* compile_params =
            dynamic_cast<kernel_selector::slice_params*>(_kernel_data.params.get());

        kernel_arguments_data args;

        args.inputs.push_back(instance.input_memory_ptr(0));

        if (compile_params->start_arg_type == kernel_selector::base_params::ArgType::Input)
            args.inputs.push_back(instance.input_memory_ptr(1));

        if (compile_params->step_arg_type == kernel_selector::base_params::ArgType::Input)
            args.inputs.push_back(instance.input_memory_ptr(3));


        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();
        return args;
    }

    static std::unique_ptr<primitive_impl> create(const slice_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::slice_params>(impl_param, impl_param.is_dynamic());
        auto op_params = get_default_optional_params<kernel_selector::slice_optional_params>(arg.get_program());
        const auto& inputs = arg.get_dependencies();
        const stream& stream = arg.get_program().get_stream();

        const auto data_shape = extractShape(params.inputs[0]);

        std::vector<std::int32_t> axes(data_shape.size());
        if (inputs.size() == InputIndices::kInputsNum)
            axes = extractIntegerData(inputs[InputIndices::kAxes].first->as<data>(), stream);
        else
            std::iota(axes.begin(), axes.end(), 0);

        if (inputs[InputIndices::kStart].first->is_constant()) {
            params.start_arg_type = kernel_selector::base_params::ArgType::Constant;
            auto elts = extractIntegerData(inputs[InputIndices::kStart].first->as<data>(), stream);
            std::vector<std::int32_t> selected_start(data_shape.size(), 0);
            for (size_t axis = 0; axis < axes.size(); axis++) {
                auto transformed_axe = axes[axis] < 0 ? data_shape.size() + axes[axis] : axes[axis];
                selected_start[transformed_axe] = elts[axis];
            }

            params.start = std::move(selected_start);

        } else {
            params.start_arg_type = kernel_selector::base_params::ArgType::Input;
            auto layout = impl_param.get_input_layout(InputIndices::kStart);
            params.inputs.push_back(convert_data_tensor(layout));
        }


        if (inputs[InputIndices::kStep].first->is_constant()) {
            params.step_arg_type = kernel_selector::base_params::ArgType::Constant;
            auto step_elts = extractIntegerData(inputs[InputIndices::kStep].first->as<data>(), stream);
            std::vector<std::int32_t> selected_step(data_shape.size(), 1);
            for (size_t axis = 0; axis < axes.size(); axis++) {
                auto transformed_axe = axes[axis] < 0 ? data_shape.size() + axes[axis] : axes[axis];
                selected_step[transformed_axe] = step_elts[axis];
            }

            params.step = std::move(selected_step);

        } else {
            params.step_arg_type = kernel_selector::base_params::ArgType::Input;
            auto stop_layout = impl_param.get_input_layout(InputIndices::kStep);
            params.inputs.push_back(convert_data_tensor(stop_layout));
        }

        // if (!inputs[InputIndices::kEnd].first->is_constant()) {
        //     auto stop_layout = impl_param.get_input_layout(InputIndices::kEnd);
        //     params.inputs.push_back(convert_data_tensor(stop_layout));
        // }

        params.set_dynamic_shape_offsets();
        auto& kernel_selector = kernel_selector::slice_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(params, op_params);

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
