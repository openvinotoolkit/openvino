// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <slice_inst.h>
#include <slice/slice_kernel_ref.h>
#include <data_inst.h>
#include <intel_gpu/runtime/error_handler.hpp>
#include <impls/implementation_map.hpp>
#include <slice/slice_kernel_selector.h>
#include "primitive_base.hpp"
#include <vector>
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
    switch (node.get_output_layout().data_type) {
    case data_types::u8:
        return extractIntegerData<std::uint8_t>(node, stream);
    case data_types::i8:
        return extractIntegerData<std::int8_t>(node, stream);
    case data_types::i32:
        return extractIntegerData<std::int32_t>(node, stream);
    case data_types::i64:
        return extractIntegerData<std::int64_t>(node, stream);
    default:
        CLDNN_ERROR_DATA_TYPES_MISMATCH(node.id(), "Slice parameter",
                node.get_output_layout().data_type, "Any integral type",
                data_types::i32, "Slice parameters should be of integral type.");
    }
    return {};
}

std::vector<std::int32_t> extractShape(kernel_selector::Tensor::DataTensor& tensor) {
    auto logical_dims = tensor.LogicalDims();
    // LogicalDims method returns dims in reversed order
    return {logical_dims.rbegin(), logical_dims.rend()};
}

} // namespace

struct slice_impl : typed_primitive_impl_ocl<slice> {
    using parent = typed_primitive_impl_ocl<slice>;
    using parent::parent;

    enum InputIndices {
        kData,
        kStart,
        kEnd,
        kStep,
        kAxes,
        kInputsNum
    };

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<slice_impl>(*this);
    }

    static primitive_impl* create(const slice_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::slice_params>(impl_param);
        auto op_params = get_default_optional_params<kernel_selector::slice_optional_params>(arg.get_program());
        const auto& inputs = arg.get_dependencies();
        const stream& stream = arg.get_program().get_stream();
        auto start_elts = extractIntegerData(inputs[InputIndices::kStart]->as<data>(), stream);
        auto end_elts = extractIntegerData(inputs[InputIndices::kEnd]->as<data>(), stream);
        auto step_elts = extractIntegerData(inputs[InputIndices::kStep]->as<data>(), stream);
        auto data_shape = extractShape(params.inputs[0]);
        std::vector<std::int32_t> axes(data_shape.size());
        if (inputs.size() == InputIndices::kInputsNum)
            axes = std::move(extractIntegerData(inputs[InputIndices::kAxes]->as<data>(), stream));
        else
            std::iota(axes.begin(), axes.end(), 0);
        std::vector<std::int32_t> selected_start(data_shape.size(), 0);
        std::vector<std::int32_t> selected_step(data_shape.size(), 1);
        std::vector<std::int32_t> selected_end(data_shape);
        for (size_t axis = 0; axis < axes.size(); axis++) {
            auto transformed_axe = axes[axis] < 0 ? data_shape.size() + axes[axis] : axes[axis];
            auto start = start_elts[axis];
            auto end = end_elts[axis];
            auto dim_size = data_shape[transformed_axe];
            selected_start[transformed_axe] = std::max(std::min(start < 0 ? dim_size + start : start, dim_size - 1), 0);
            selected_end[transformed_axe] = std::max(std::min(end < 0 ? dim_size + end : end, dim_size - 1), 0);
            selected_step[transformed_axe] = step_elts[axis];
        }
        params.start = std::move(selected_start);
        params.end = std::move(selected_end);
        params.step = std::move(selected_step);
        auto &kernel_selector =
                kernel_selector::slice_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, op_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(),
                "Cannot find a proper kernel with this arguments");

        return new slice_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_slice_impl::attach_slice_impl() {
    implementation_map<slice>::add(impl_types::ocl, slice_impl::create, {
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),
    });
}

}  // namespace detail

} // namespace ocl
} // namespace cldnn
