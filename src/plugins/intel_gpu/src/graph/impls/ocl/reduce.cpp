// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "reduce_inst.h"
#include "reduce/reduce_kernel_selector.h"
#include "reduce/reduce_kernel_ref.h"

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

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<reduce>();
        auto params = get_default_params<kernel_selector::reduce_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::reduce_optional_params>(impl_param.get_program());

        params.reduceAxes = convert_axes(primitive->axes, impl_param.input_layouts[0].get_rank());
        params.keepDims = primitive->keep_dims;
        params.reduceMode = cldnn_2_reduce_mode(primitive->mode);

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
        update_kernels_list_to_skip();
    }
};

namespace detail {

attach_reduce_impl::attach_reduce_impl() {
    auto types = {
        data_types::f32,
        data_types::f16,
        data_types::i32,
        data_types::i8,
        data_types::u8
    };

    auto static_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv16
    };

    implementation_map<reduce>::add(impl_types::ocl,
                                    shape_types::static_shape,
                                    typed_primitive_impl_ocl<reduce>::create<reduce_impl>,
                                    types,
                                    static_formats);

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<reduce>::add(impl_types::ocl,
                                    shape_types::dynamic_shape,
                                    typed_primitive_impl_ocl<reduce>::create<reduce_impl>,
                                    types,
                                    dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reduce_impl)
