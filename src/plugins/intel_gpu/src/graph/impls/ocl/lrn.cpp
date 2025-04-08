// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "lrn_inst.h"
#include "lrn/lrn_kernel_selector.h"
#include "lrn/lrn_kernel_base.h"

namespace cldnn {
namespace ocl {

struct lrn_impl : typed_primitive_impl_ocl<lrn> {
    using parent = typed_primitive_impl_ocl<lrn>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lrn_kernel_selector;
    using kernel_params_t = kernel_selector::lrn_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::lrn_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<lrn_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<lrn>();
        auto params = get_default_params<kernel_selector::lrn_params>(impl_param);

        params.alpha = primitive->alpha;
        params.beta = primitive->beta;
        params.k = primitive->k;
        params.localSize = primitive->size;
        params.divMode = kernel_selector::kernel_divider_mode::FIXED;
        params.normMode = primitive->norm_region == lrn_norm_region_within_channel
                                  ? kernel_selector::lrn_mode::WITHIN_CHANNEL
                                  : kernel_selector::lrn_mode::ACROSS_CHANNEL;

        return params;
    }
};

namespace detail {

attach_lrn_impl::attach_lrn_impl() {
    implementation_map<lrn>::add(impl_types::ocl, typed_primitive_impl_ocl<lrn>::create<lrn_impl>, {
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),

        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),

        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv4),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv4),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv16),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv16),

        std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::i8, format::b_fs_yx_fsv32),
        std::make_tuple(data_types::u8, format::b_fs_yx_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv32),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv32),

        std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::i8, format::bs_fs_yx_bsv32_fsv16),
        std::make_tuple(data_types::u8, format::bs_fs_yx_bsv32_fsv16),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lrn_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lrn)
