// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "shuffle_channels_inst.h"
#include "shuffle_channels/shuffle_channels_kernel_selector.h"
#include "shuffle_channels/shuffle_channels_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct shuffle_channels_impl : typed_primitive_impl_ocl<shuffle_channels> {
    using parent = typed_primitive_impl_ocl<shuffle_channels>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::shuffle_channels_kernel_selector;
    using kernel_params_t = kernel_selector::shuffle_channels_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::shuffle_channels_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<shuffle_channels_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<shuffle_channels>();
        auto params = get_default_params<kernel_selector::shuffle_channels_params>(impl_param);

        const int32_t number_of_dims = 4;
        int32_t axis = primitive->axis;

        if (axis < 0)
            axis += number_of_dims;

        params.group = primitive->group;
        params.axis = axis;

        return params;
    }
};

namespace detail {

attach_shuffle_channels_impl::attach_shuffle_channels_impl() {
    auto types =
        {data_types::f16, data_types::f32, data_types::i8, data_types::u8};
    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv4,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::fs_b_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
    };

    implementation_map<shuffle_channels>::add(impl_types::ocl, typed_primitive_impl_ocl<shuffle_channels>::create<shuffle_channels_impl>, types, formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::shuffle_channels_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::shuffle_channels)
