// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reduce/reduce_kernel_selector.h"
#include "reduce/reduce_kernel_ref.h"
#include "reduce/reduce_kernel_b_fs_yx_fsv16.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "data_inst.h"

using namespace cldnn;

namespace cldnn {
namespace ocl {
namespace {
static std::vector<uint16_t> convert_axes(std::vector<int64_t> axes, size_t rank) {
    std::vector<uint16_t> converted_axes;
    for (auto axis : axes) {
        if (axis == 0 || axis == 1) {
            converted_axes.push_back(axis);
            continue;
        }

        if (axis < 0)
            axis = axis + rank;

        converted_axes.push_back(rank + 1 - axis);
    }

    return converted_axes;
}

kernel_selector::reduce_mode cldnn_2_reduce_mode(reduce_mode mode) {
    switch (mode) {
        case reduce_mode::max:
            return kernel_selector::reduce_mode::MAX;
        case reduce_mode::min:
            return kernel_selector::reduce_mode::MIN;
        case reduce_mode::mean:
            return kernel_selector::reduce_mode::MEAN;
        case reduce_mode::prod:
            return kernel_selector::reduce_mode::PROD;
        case reduce_mode::sum:
            return kernel_selector::reduce_mode::SUM;
        case reduce_mode::logical_and:
            return kernel_selector::reduce_mode::AND;
        case reduce_mode::logical_or:
            return kernel_selector::reduce_mode::OR;
        case reduce_mode::sum_square:
            return kernel_selector::reduce_mode::SUM_SQUARE;
        case reduce_mode::l1:
            return kernel_selector::reduce_mode::L1;
        case reduce_mode::l2:
            return kernel_selector::reduce_mode::L2;
        case reduce_mode::log_sum:
            return kernel_selector::reduce_mode::LOG_SUM;
        case reduce_mode::log_sum_exp:
            return kernel_selector::reduce_mode::LOG_SUM_EXP;
        default:
            assert(0);
            return kernel_selector::reduce_mode::MAX;
    }
}
}  // namespace
struct reduce_impl : typed_primitive_impl_ocl<reduce> {
    using parent = typed_primitive_impl_ocl<reduce>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reduce_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::reduce_params, kernel_selector::reduce_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reduce_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<reduce>();
        auto params = get_default_params<kernel_selector::reduce_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::reduce_optional_params>(impl_param.get_program());

        params.reduceAxes = convert_axes(primitive->axes, impl_param.input_layouts[0].get_rank());
        params.keepDims = primitive->keep_dims;
        params.reduceMode = cldnn_2_reduce_mode(primitive->mode);

        return {params, optional_params};
    }
};

namespace detail {

attach_reduce_impl::attach_reduce_impl() {
    implementation_map<reduce>::add(impl_types::ocl, typed_primitive_impl_ocl<reduce>::create<reduce_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),

        std::make_tuple(data_types::f32, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i8, format::bfwzyx),
        std::make_tuple(data_types::u8, format::bfwzyx),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reduce_impl)
