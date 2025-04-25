// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>

#include "data_inst.h"
#include "primitive_base.hpp"
#include "slice/slice_kernel_ref.h"
#include "slice/slice_kernel_selector.h"
#include "slice_inst.h"

namespace cldnn {
namespace ocl {

namespace {
template <typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
std::vector<std::int64_t> extractIntegerData(const data_node& node, const stream& stream) {
    mem_lock<T> lock{node.get_attached_memory_ptr(), stream};
    T* data = lock.data();
    std::vector<std::int64_t> integer_data;
    integer_data.reserve(node.get_output_layout().count());
    for (size_t i = 0; i < node.get_output_layout().count(); i++) {
        integer_data.emplace_back(static_cast<std::int64_t>(data[i]));
    }
    return integer_data;
}

std::vector<std::int64_t> extractIntegerData(const data_node& node, const stream& stream) {
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
        OPENVINO_ASSERT(false,
                        "[GPU] Slice parameters should be of integral type for node ",
                        node.id(),
                        " while got ",
                        dt);
    }
    return {};
}

}  // namespace

struct slice_impl : typed_primitive_impl_ocl<slice> {
    using parent = typed_primitive_impl_ocl<slice>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::slice_kernel_selector;
    using kernel_params_t = kernel_selector::slice_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::slice_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<slice_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    kernel_arguments_data get_arguments(const slice_inst& instance) const override {
        kernel_arguments_data args;

        const SliceKernelRefNeededInputs inputs = SliceKernelRefNeededInputs::Create(*instance.node);

        for (auto idx : inputs.GetNeededInputIndexes()) {
            args.inputs.push_back(instance.input_memory_ptr(idx));
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }

        args.shape_info = instance.shape_info_memory_ptr();
        return args;
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::slice_params>(impl_param, is_shape_agnostic);
        const auto input_rank = params.inputs[0].Dimentions();
        const auto& arg = impl_param.prog->get_node(impl_param.desc->id);

        if (!PrepareInput(arg,
                          SliceKernelRefNeededInputs::kStart,
                          params.compile_time_start,
                          params.start_data_type,
                          params.inputs)) {
            // No kStart input - set it to default:
            params.axes_data_type = kernel_selector::Datatype::INT64;
            params.compile_time_start = std::vector<int64_t>(input_rank, 0);
        }

        // NOTE: Stop input is not used by the slice kernel, as this information
        // is implicitely passed with output shape.

        if (!PrepareInput(arg,
                          SliceKernelRefNeededInputs::kStep,
                          params.compile_time_step,
                          params.step_data_type,
                          params.inputs)) {
            // No kStep input - set it to default:
            params.axes_data_type = kernel_selector::Datatype::INT64;
            params.compile_time_step = std::vector<int64_t>(input_rank, 1);
        }

        if (!PrepareInput(arg,
                          SliceKernelRefNeededInputs::kAxes,
                          params.compile_time_axes,
                          params.axes_data_type,
                          params.inputs)) {
            // No kAxes input - set it to default:
            params.axes_data_type = kernel_selector::Datatype::INT64;
            params.compile_time_axes.resize(input_rank);
            std::iota(params.compile_time_axes.begin(), params.compile_time_axes.end(), 0);
        }

        // Transform compile time axes:
        for (size_t axis = 0; axis < params.compile_time_axes.size(); ++axis) {
            const int64_t transformed_axe = params.compile_time_axes[axis] < 0
                                             ? input_rank + params.compile_time_axes[axis]
                                             : params.compile_time_axes[axis];
            params.compile_time_axes[axis] = transformed_axe;
        }

        params.set_dynamic_shape_offsets();

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }

private:
    // Returns true if input was prepared(was avaiable in node def), false otherwise.
    static bool PrepareInput(const slice_node& arg,
                             SliceKernelRefNeededInputs::InputIndices idx,
                             std::vector<std::int64_t>& out_compile_time_buff,
                             kernel_selector::Datatype& out_buff_data_type,
                             kernel_selector::MultiDataTensor& out_runtime_inputs) {
        const stream& stream = arg.get_program().get_stream();
        const auto& inputs = arg.get_dependencies();

        if (inputs.size() <= idx)
            return false;

        const SliceKernelRefNeededInputs kernel_needed_inputs = SliceKernelRefNeededInputs::Create(arg);
        if (kernel_needed_inputs.IsInputNeededInRuntime(idx)) {
            const auto layout = inputs[idx].first->get_output_layout(0);
            out_buff_data_type = to_data_type(layout.data_type);
            out_compile_time_buff.clear();
            out_runtime_inputs.push_back(convert_data_tensor(layout));
        } else {
            out_buff_data_type = kernel_selector::Datatype::INT64;
            out_compile_time_buff = extractIntegerData(inputs[idx].first->as<data>(), stream);
        }

        return true;
    }
};

namespace detail {

attach_slice_impl::attach_slice_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32,
        data_types::i64
    };

    auto formats = {
        format::bfyx,
        format::bfzyx,
    };

    implementation_map<slice>::add(impl_types::ocl,
                                   shape_types::any,
                                   typed_primitive_impl_ocl<slice>::create<slice_impl>,
                                   types,
                                   formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::slice_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::slice)
