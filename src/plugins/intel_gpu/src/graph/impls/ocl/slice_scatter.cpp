// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>

#include "data_inst.h"
#include "primitive_base.hpp"
#include "slice_scatter/slice_scatter_kernel_ref.h"
#include "slice_scatter/slice_scatter_kernel_selector.h"
#include "slice_scatter_inst.h"

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
                        "[GPU] SliceScatter parameters should be of integral type for node ",
                        node.id(),
                        " while got ",
                        dt);
    }
    return {};
}

}  // namespace

struct slice_scatter_impl : typed_primitive_impl_ocl<slice_scatter> {
    using parent = typed_primitive_impl_ocl<slice_scatter>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::slice_scatter_kernel_selector;
    using kernel_params_t = kernel_selector::slice_scatter_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::slice_scatter_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<slice_scatter_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    kernel_arguments_data get_arguments(const slice_scatter_inst& instance) const override {
        kernel_arguments_data args;

        const SliceScatterKernelRefNeededInputs inputs = SliceScatterKernelRefNeededInputs::Create(instance.get_node());

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
        auto params = get_default_params<kernel_selector::slice_scatter_params>(impl_param, is_shape_agnostic);
        const auto input_rank = params.inputs[0].Dimentions();
        const auto& arg = impl_param.prog->get_node(impl_param.desc->id);

        // Add updates tensor (INPUT1) - it's always present and needed in runtime
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));

        if (!PrepareInput(arg,
                          SliceScatterKernelRefNeededInputs::kStart,
                          params.compile_time_start,
                          params.start_data_type,
                          params.inputs)) {
            params.start_data_type = kernel_selector::Datatype::INT64;
            params.compile_time_start = std::vector<int64_t>(input_rank, 0);
        }

        // NOTE: Stop input is not used by the kernel, as information
        // is implicitly passed with updates shape.

        if (!PrepareInput(arg,
                          SliceScatterKernelRefNeededInputs::kStep,
                          params.compile_time_step,
                          params.step_data_type,
                          params.inputs)) {
            params.step_data_type = kernel_selector::Datatype::INT64;
            params.compile_time_step = std::vector<int64_t>(input_rank, 1);
        }

        if (!PrepareInput(arg,
                          SliceScatterKernelRefNeededInputs::kAxes,
                          params.compile_time_axes,
                          params.axes_data_type,
                          params.inputs)) {
            params.axes_data_type = kernel_selector::Datatype::INT64;
            params.compile_time_axes.resize(input_rank);
            std::iota(params.compile_time_axes.begin(), params.compile_time_axes.end(), 0);
        }

        // Transform compile time axes:
        for (size_t axis = 0; axis < params.compile_time_axes.size(); ++axis) {
            const int64_t transformed_axis = params.compile_time_axes[axis] < 0
                                              ? input_rank + params.compile_time_axes[axis]
                                              : params.compile_time_axes[axis];
            params.compile_time_axes[axis] = transformed_axis;
        }

        params.set_dynamic_shape_offsets();

        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }

private:
    static bool PrepareInput(const slice_scatter_node& arg,
                             SliceScatterKernelRefNeededInputs::InputIndices idx,
                             std::vector<std::int64_t>& out_compile_time_buff,
                             kernel_selector::Datatype& out_buff_data_type,
                             kernel_selector::MultiDataTensor& out_runtime_inputs) {
        const stream& stream = arg.get_program().get_stream();
        const auto& inputs = arg.get_dependencies();

        if (inputs.size() <= idx)
            return false;

        const SliceScatterKernelRefNeededInputs kernel_needed_inputs = SliceScatterKernelRefNeededInputs::Create(arg);
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

attach_slice_scatter_impl::attach_slice_scatter_impl() {
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

    implementation_map<slice_scatter>::add(impl_types::ocl,
                                           shape_types::any,
                                           typed_primitive_impl_ocl<slice_scatter>::create<slice_scatter_impl>,
                                           types,
                                           formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::slice_scatter_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::slice_scatter)
