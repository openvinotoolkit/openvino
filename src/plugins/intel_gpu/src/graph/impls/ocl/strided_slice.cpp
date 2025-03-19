// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "strided_slice_inst.h"
#include "data_inst.h"
#include "strided_slice/strided_slice_kernel_ref.h"
#include "strided_slice/strided_slice_kernel_selector.h"


namespace {
template <typename T, typename DT, typename = typename std::enable_if<std::is_convertible<DT, T>::value>::type>
void pad_vector_to_size(std::vector<T>& data, size_t size, DT value, const std::vector<int64_t>& ellipsis_mask) {
    bool apply_ellipsis_mask = std::count(ellipsis_mask.begin(), ellipsis_mask.end(), 1) == 1;
    if (apply_ellipsis_mask && data.size() == ellipsis_mask.size()) {
        std::vector<T> temp;
        size_t ellipsis_pos1 = 0;
        for (size_t i = 0; i < ellipsis_mask.size(); i++) {
            if (ellipsis_mask[i] == 1) {
                ellipsis_pos1 = i;
                break;
            }
        }

        size_t dims_after = data.size() - ellipsis_pos1 - 1;
        size_t ellipsis_pos2 = size - dims_after - 1;;

        for (size_t i = 0; i < ellipsis_pos1; i++)
            temp.push_back(data[i]);

        for (size_t i = ellipsis_pos1; i < ellipsis_pos2 + 1; i++)
            temp.push_back(value);

        for (size_t i = 1; i < size - ellipsis_pos2; i++)
            temp.push_back(data[i + ellipsis_pos1]);

        data = temp;
    } else {
        for (size_t i = data.size(); i < size; ++i) {
            data.push_back(static_cast<T>(value));
        }
    }
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
    using kernel_params_t = kernel_selector::strided_slice_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::strided_slice_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<strided_slice_impl, kernel_params_t>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& prim = impl_param.typed_desc<strided_slice>();
        auto params = get_default_params<kernel_selector::strided_slice_params>(impl_param, is_shape_agnostic);
        const size_t dims_num = params.inputs[0].Dimentions();

        std::vector<int32_t> begin(prim->begin.begin(), prim->begin.end());
        std::vector<int32_t> end(prim->end.begin(), prim->end.end());
        std::vector<int32_t> strides(prim->strides.begin(), prim->strides.end());

        // Getting data from constant inputs. There are 3 args: Begin, End, Stride
        if (!begin.empty() && !params.has_dynamic_tensors()) {
            pad_vector_to_size(begin, dims_num, 0, prim->ellipsis_mask);
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
            pad_vector_to_size(end, dims_num, 1, prim->ellipsis_mask);
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
            pad_vector_to_size(strides, dims_num, 1, prim->ellipsis_mask);
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
        auto ellipsis_mask_ = prim->ellipsis_mask;

        std::vector<uint8_t> begin_mask(begin_mask_.begin(), begin_mask_.end());
        std::vector<uint8_t> end_mask(end_mask_.begin(), end_mask_.end());
        std::vector<uint8_t> new_axis_mask(new_axis_mask_.begin(), new_axis_mask_.end());
        std::vector<uint8_t> shrink_axis_mask(shrink_axis_mask_.begin(), shrink_axis_mask_.end());
        std::vector<uint8_t> ellipsis_mask(ellipsis_mask_.begin(), ellipsis_mask_.end());
        params.end_mask = std::move(end_mask);
        pad_vector_to_size(params.end_mask, dims_num, 0, prim->ellipsis_mask);
        params.begin_mask = std::move(begin_mask);
        pad_vector_to_size(params.begin_mask, dims_num, 0, prim->ellipsis_mask);

        params.new_axis_mask = new_axis_mask;
        params.shrink_axis_mask = shrink_axis_mask;
        params.ellipsis_mask = ellipsis_mask;
        pad_vector_to_size(params.shrink_axis_mask, dims_num, 0, prim->ellipsis_mask);

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
                auto begin = params.striding_params[0][dim];
                auto end = params.striding_params[1][dim];
                auto stride = params.striding_params[2][dim];

                // Check out of bounds values for Clamping
                auto check_out_of_bounds = [&](int32_t value) -> bool {
                    auto size = out_shape[dim];
                    if (value >= size || value < (size * -1))
                        return true;
                    else
                        return false;
                };
                bool should_clamp_begin = check_out_of_bounds(begin);
                bool should_clamp_end = check_out_of_bounds(end);

                // Convert a negative value which means reverse indexing from the end
                if (begin < 0)
                    begin += out_shape[dim];  // converted value can be negative if the original one was out of bounds
                if (end < 0)
                    end += out_shape[dim];
                bool is_stride_reverse = (stride < 0) ? true : false;

                // Clamping
                begin = std::min(std::max(begin, (int32_t)0), out_shape[dim]);
                end = std::min(std::max(end, (int32_t)0), out_shape[dim]);

                if (is_stride_reverse) {
                    // If begin > end && is_reverse, then we don't need to adjust begin/end values, the kernel will process it correctly
                    // However, in case of out-of-bounds begin/end values, it will be clamped, so we subtract 1 from each of them manually
                    // E.g. out_shape[dim] = 100; begin=10000; end=-10000; stride=-1
                    // clamp: begin=100; end=0;
                    // sub: begin=99; end=-1;
                    // If begin <= end, then we swap begin/end values and subtruct 1 from each of them
                    // E.g. out_shape[dim] = 100; begin=-100; end=100; stride=-1
                    // sub: begin=-1; end=100;
                    // swap: begin=100; end=-1;
                    // So the kernel will put the slices [99, 0] in reversed order as expected.
                    if (should_clamp_begin)
                        begin--;
                    if (should_clamp_end)
                        end--;
                    if (begin <= end)
                        std::swap(begin, end);
                }

                params.striding_params[0][dim] = begin;
                params.striding_params[1][dim] = end;
            }
        }
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
