// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "strided_slice_inst.h"
#include "data_inst.h"
#include "strided_slice/strided_slice_kernel_ref.h"
#include "strided_slice/strided_slice_kernel_selector.h"

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
        if (mask[i])
            dst[i] = src;
    }
    return dst;
}

template <typename T, typename MT>
std::vector<T>& vector_assign_if_not_mask(std::vector<T>& dst, const std::vector<T>& src, const std::vector<MT>& mask) {
    for (size_t i = 0; i < dst.size(); ++i) {
        if (mask[i])
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

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::strided_slice_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<strided_slice_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->SetUpdateDispatchDataFunc(_kernel_data);
        }
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& prim = impl_param.typed_desc<strided_slice>();
        auto params = get_default_params<kernel_selector::strided_slice_params>(impl_param, is_shape_agnostic);
        auto op_params = get_default_optional_params<kernel_selector::strided_slice_optional_params>(impl_param.get_program());
        const size_t dims_num = params.inputs[0].Dimentions();

        std::vector<int32_t> begin(prim->begin.begin(), prim->begin.end());
        std::vector<int32_t> end(prim->end.begin(), prim->end.end());
        std::vector<int32_t> strides(prim->strides.begin(), prim->strides.end());

        // Getting data from constant inputs. There are 3 args: Begin, End, Stride
        if (!begin.empty() && !params.has_dynamic_tensors()) {
            pad_vector_to_size(begin, dims_num, 0);
            params.begin_type = kernel_selector::base_params::ArgType::Constant;
            params.striding_params.push_back(begin);
        } else {
            params.begin_type = kernel_selector::base_params::ArgType::Input;
            auto begin_layout = impl_param.get_input_layout(1);
            params.inputs.push_back(convert_data_tensor(begin_layout));
            params.begin_dims = begin_layout.count();
        }

        auto get_index_end = [&]() {
            size_t offset = 1;
            if ((begin.empty() || params.has_dynamic_tensors()) && params.begin_type == kernel_selector::base_params::ArgType::Input)
                offset++;
            return offset;
        };
        if (!end.empty() && !params.has_dynamic_tensors()) {
            pad_vector_to_size(end, dims_num, 1);
            params.end_type = kernel_selector::base_params::ArgType::Constant;
            params.striding_params.push_back(end);
        } else {
            params.end_type = kernel_selector::base_params::ArgType::Input;
            auto end_layout = impl_param.get_input_layout(get_index_end());
            params.inputs.push_back(convert_data_tensor(end_layout));
            params.end_dims = end_layout.count();
        }

        auto get_index_stride = [&]() {
            size_t offset = get_index_end();
            if ((end.empty() || params.has_dynamic_tensors()) && params.end_type == kernel_selector::base_params::ArgType::Input)
                offset++;
            return offset;
        };
        if (!strides.empty() && !params.has_dynamic_tensors()) {
            pad_vector_to_size(strides, dims_num, 1);
            params.stride_type = kernel_selector::base_params::ArgType::Constant;
            params.striding_params.push_back(strides);
        } else {
            params.stride_type = kernel_selector::base_params::ArgType::Input;
            auto stride_layout = impl_param.get_input_layout(get_index_stride());
            params.inputs.push_back(convert_data_tensor(stride_layout));
            params.stride_dims = stride_layout.count();
        }

        auto begin_mask_ = prim->begin_mask;
        auto end_mask_ = prim->end_mask;
        auto new_axis_mask_ = prim->new_axis_mask;
        auto shrink_axis_mask_ = prim->shrink_axis_mask;

        std::vector<uint8_t> begin_mask(begin_mask_.begin(), begin_mask_.end());
        std::vector<uint8_t> end_mask(end_mask_.begin(), end_mask_.end());
        std::vector<uint8_t> new_axis_mask(new_axis_mask_.begin(), new_axis_mask_.end());
        std::vector<uint8_t> shrink_axis_mask(shrink_axis_mask_.begin(), shrink_axis_mask_.end());
        params.end_mask = std::move(end_mask);
        pad_vector_to_size(params.end_mask, dims_num, 0);
        params.begin_mask = std::move(begin_mask);
        pad_vector_to_size(params.begin_mask, dims_num, 0);

        params.new_axis_mask = new_axis_mask;
        params.shrink_axis_mask = shrink_axis_mask;
        pad_vector_to_size(params.shrink_axis_mask, dims_num, 0);

        std::vector<size_t> logical_dims = params.inputs[0].LogicalDims();
        std::reverse(logical_dims.begin(), logical_dims.end());  // get dims in bfyx order
        std::vector<int32_t> out_shape;
        for (const auto& dim : logical_dims)
            out_shape.push_back(static_cast<int32_t>(dim));
        if (params.striding_params.size() == 3) {
            // If the ith bit of begin_mask is not set, begin[i] is ignored and the range of the appropriate dimension starts from 0.
            vector_assign_if_not_mask(params.striding_params[0], 0, params.begin_mask);
            // If the ith bit of end_mask is not set, end[i] is ignored and the fullest possible range in that dimension is used
            // instead.
            vector_assign_if_not_mask(params.striding_params[1], out_shape, params.end_mask);
            for (size_t dim = 0; dim < params.striding_params[2].size(); dim++) {
                auto begin_org = params.striding_params[0][dim];
                auto end_org = params.striding_params[1][dim];
                if (params.striding_params[0][dim] < 0)
                    params.striding_params[0][dim] = std::max(out_shape[dim] + params.striding_params[0][dim], (int32_t)0);
                if (params.striding_params[1][dim] < 0)
                    params.striding_params[1][dim] = std::max(out_shape[dim] + params.striding_params[1][dim], (int32_t)0);

                params.striding_params[0][dim] = std::min(params.striding_params[0][dim], out_shape[dim]);
                params.striding_params[1][dim] = std::min(params.striding_params[1][dim], out_shape[dim]);

                auto& begin = params.striding_params[0][dim];
                auto& end = params.striding_params[1][dim];
                auto& stride = params.striding_params[2][dim];
                bool is_clamp_begin = begin_org != begin;
                bool is_clamp_end = end_org != end;
                bool is_reverse = stride < 0;
                // If begin > end && is_reverse, then we don't need to adjust begin/end values, the kernel will process it correctly
                // However, in case of out-of-bounds begin/end values, it will be clamped, so we subtract 1 from each of them manually
                // E.g. out_shape[dim] = 100; begin=10000; end=-10000; stride=-1
                // clamp: begin=100; end=0;
                // sub: begin=99; end=-1;
                // If begin <= end, then we swap begin/end values and subtruct 1 from each of them
                // E.g. out_shape[dim] = 100; begin=0; end=100; stride=-1
                // swap: begin=100; end=0;
                // sub: begin=99; end=-1;
                // So the kernel will put the slices [99, 0] in reversed order as expected.
                if (is_reverse) {
                    if (begin <= end) {
                        std::swap(begin, end);
                        begin--;
                        end--;
                    } else if (begin_org != -1) {  // If begin is -1 with negative stride, clamping begin is already expected value
                        if (is_clamp_begin)
                            begin--;
                        if (is_clamp_end)
                            end--;
                    }
                }
            }
        }
        return {params, op_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_strided_slice_impl::attach_strided_slice_impl() {
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
        format::bfwzyx,
    };

    implementation_map<strided_slice>::add(impl_types::ocl,
                                           shape_types::static_shape,
                                           typed_primitive_impl_ocl<strided_slice>::create<strided_slice_impl>,
                                           types,
                                           formats);

    implementation_map<strided_slice>::add(impl_types::ocl,
                                           shape_types::dynamic_shape,
                                           typed_primitive_impl_ocl<strided_slice>::create<strided_slice_impl>,
                                           types,
                                           formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::strided_slice_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::strided_slice)
