// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "strided_slice/strided_slice_kernel_ref.h"
#include "strided_slice/strided_slice_kernel_selector.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "data_inst.h"
#include <vector>

using namespace cldnn;

namespace {
template <typename T, typename DT, typename = typename std::enable_if<std::is_convertible<DT, T>::value>::type>
std::vector<T>& pad_vector_to_size(std::vector<T>& data, size_t size, DT value) {
    for (size_t i = data.size(); i < size; ++i) {
        data.push_back(static_cast<T>(value));
    }
    return data;
}

template <typename T, typename MT>
std::vector<T>& vector_assign_if_not_mask(std::vector<T>& dst, const T& src, const std::vector<MT>& mask) {
    for (size_t i = 0; i < dst.size(); ++i) {
        if (!mask[i])
            dst[i] = src;
    }
    return dst;
}

template <typename T, typename MT>
std::vector<T>& vector_assign_if_not_mask(std::vector<T>& dst, const std::vector<T>& src, const std::vector<MT>& mask) {
    for (size_t i = 0; i < dst.size(); ++i) {
        if (!mask[i])
            dst[i] = src[i];
    }
    return dst;
}
}  // namespace

namespace cldnn {
namespace ocl {

struct strided_slice_impl : typed_primitive_impl_ocl<strided_slice> {
    using parent = typed_primitive_impl_ocl<strided_slice>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::strided_slice_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::strided_slice_params, kernel_selector::strided_slice_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<strided_slice_impl>(*this);
    }

public:
    static std::unique_ptr<primitive_impl> create(const strided_slice_node& arg, const kernel_impl_params& impl_param) {
        const auto& prim = impl_param.typed_desc<strided_slice>();
        auto params = get_default_params<kernel_selector::strided_slice_params>(impl_param);
        auto op_params = get_default_optional_params<kernel_selector::strided_slice_optional_params>(impl_param.get_program());
        const size_t dims_num = params.inputs[0].Dimentions();

        std::vector<int32_t> begin(prim->begin.begin(), prim->begin.end());
        std::vector<int32_t> end(prim->end.begin(), prim->end.end());
        std::vector<int32_t> strides(prim->strides.begin(), prim->strides.end());
        // Getting data from constant inputs. There are 3 args: Begin, End, Stride
        if (!begin.empty() && !end.empty() && !strides.empty()) {
            pad_vector_to_size(begin, dims_num, 0);
            params.striding_params.push_back(begin);
            pad_vector_to_size(end, dims_num, 1);
            params.striding_params.push_back(end);
            pad_vector_to_size(strides, dims_num, 1);
            params.striding_params.push_back(strides);
        } else {
            for (size_t i = 1; i < arg.get_dependencies().size(); ++i) {
                OPENVINO_ASSERT(impl_param.memory_deps.count(i) > 0, "[GPU] Can't find StridedSlice memory dependency");
                auto mem = impl_param.memory_deps.at(i);
                std::vector<int32_t> sizes = read_vector<int32_t>(mem, impl_param.prog->get_stream());
                pad_vector_to_size(sizes, dims_num, i != 1);  // for "begin" completion used 0 value, for other - 1
                params.striding_params.push_back(sizes);
            }
        }

        auto begin_mask_ = prim->begin_mask;
        auto end_mask_ = prim->end_mask;
        auto new_axis_mask_ = prim->new_axis_mask;
        auto shrink_axis_mask_ = prim->shrink_axis_mask;

        std::vector<uint8_t> begin_mask(begin_mask_.begin(), begin_mask_.end());
        std::vector<uint8_t> end_mask(end_mask_.begin(), end_mask_.end());
        std::vector<uint8_t> new_axis_mask(new_axis_mask_.begin(), new_axis_mask_.end());
        std::vector<uint8_t> shrink_axis_mask(shrink_axis_mask_.begin(), shrink_axis_mask_.end());
        // Plugin requires inverted mask values. Consider changing primitive impl to be aligned with the spec.
        for (auto& b : begin_mask) {
            b = 1 - b;
        }
        for (auto& e : end_mask) {
            e = 1 - e;
        }
        params.end_mask = end_mask;
        pad_vector_to_size(params.end_mask, dims_num, 1);
        params.begin_mask = begin_mask;
        pad_vector_to_size(params.begin_mask, dims_num, 1);

        params.new_axis_mask = new_axis_mask;
        params.shrink_axis_mask = shrink_axis_mask;
        pad_vector_to_size(params.shrink_axis_mask, dims_num, 0);

        std::vector<size_t> logical_dims = params.inputs[0].LogicalDims();
        std::reverse(logical_dims.begin(), logical_dims.end());  // get dims in bfyx order
        std::vector<int32_t> out_shape;
        for (const auto& dim : logical_dims)
            out_shape.push_back(static_cast<int32_t>(dim));
        // If the ith bit of begin_mask is not set, begin[i] is ignored and the range of the appropriate dimension starts from 0.
        vector_assign_if_not_mask(params.striding_params[0], 0, params.begin_mask);
        // If the ith bit of end_mask is not set, end[i] is ignored and the fullest possible range in that dimension is used
        // instead.
        vector_assign_if_not_mask(params.striding_params[1], out_shape, params.end_mask);
        for (size_t dim = 0; dim < params.striding_params[2].size(); dim++) {
            if (params.striding_params[0][dim] < 0)
                params.striding_params[0][dim] = std::max(out_shape[dim] + params.striding_params[0][dim], (int32_t)0);
            if (params.striding_params[1][dim] < 0)
                params.striding_params[1][dim] = std::max(out_shape[dim] + params.striding_params[1][dim], (int32_t)0);

            params.striding_params[0][dim] = std::min(params.striding_params[0][dim], out_shape[dim]);
            params.striding_params[1][dim] = std::min(params.striding_params[1][dim], out_shape[dim]);

            auto& begin = params.striding_params[0][dim];
            auto& end = params.striding_params[1][dim];
            auto& stride = params.striding_params[2][dim];
            bool is_reverse = stride < 0;
            // If begin > end && is_reverse, then we don't need to adjust begin/end values, the kernel will process it correctly
            // If begin <= end, then we swap begin/end values and subtruct 1 from each of them
            // E.g. out_shape[dim] = 100; begin=0; end=100; stride=-1
            // swap: begin=100; end=0;
            // sub: begin=99; end=-1;
            // So the kernel will put the slices [99, 0] in reversed order as expected.
            if (is_reverse && begin <= end) {
                std::swap(begin, end);
                begin--;
                end--;
            }
        }

        auto& kernel_selector = kernel_selector::strided_slice_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(params, op_params);

        return make_unique<strided_slice_impl>(arg, best_kernel);
    }
};

namespace detail {

attach_strided_slice_impl::attach_strided_slice_impl() {
    implementation_map<strided_slice>::add(impl_types::ocl, strided_slice_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::strided_slice_impl)
