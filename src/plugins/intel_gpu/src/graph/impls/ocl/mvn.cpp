// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "mvn_inst.h"
#include "mvn/mvn_kernel_selector.h"
#include "mvn/mvn_kernel_base.h"

namespace cldnn {
namespace ocl {

struct mvn_impl : typed_primitive_impl_ocl<mvn> {
    using parent = typed_primitive_impl_ocl<mvn>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::mvn_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::mvn_params, kernel_selector::mvn_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mvn_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<mvn>();
        auto params = get_default_params<kernel_selector::mvn_params>(impl_param, is_shape_agnostic);
        auto optional_params = get_default_optional_params<kernel_selector::mvn_optional_params>(impl_param.get_program());

        params.mvnMode = primitive->across_channels ? kernel_selector::mvn_mode::ACROSS_CHANNELS
                                                    : kernel_selector::mvn_mode::WITHIN_CHANNELS;
        params.mvnNormalizeVariance = primitive->normalize_variance;
        params.epsilon = primitive->epsilon;

        params.mvnEpsMode = primitive->eps_inside_sqrt ? kernel_selector::mvn_eps_mode::INSIDE_SQRT
                                                       : kernel_selector::mvn_eps_mode::OUTSIDE_SQRT;
        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
        update_kernels_list_to_skip();
    }
};

namespace detail {

attach_mvn_impl::attach_mvn_impl() {
    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx
    };

    implementation_map<mvn>::add(impl_types::ocl,
                                 shape_types::dynamic_shape,
                                 typed_primitive_impl_ocl<mvn>::create<mvn_impl>,
                                 dyn_types,
                                 dyn_formats);

    implementation_map<mvn>::add(impl_types::ocl, typed_primitive_impl_ocl<mvn>::create<mvn_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),

        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::u8, format::bfzyx),
        std::make_tuple(data_types::i8, format::bfzyx),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv16),

        std::make_tuple(data_types::f32, format::bs_fs_zyx_bsv16_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_zyx_bsv16_fsv16),

        // TODO: uncomment this code when fsv32 optimizations for MVN will be implemented
        /*std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_zyx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_zyx_fsv32),*/

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::mvn_impl)
