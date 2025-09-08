// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "eye_inst.h"
#include "eye/eye_kernel_ref.h"
#include "eye/eye_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct eye_impl : typed_primitive_impl_ocl<eye> {
    using parent = typed_primitive_impl_ocl<eye>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::eye_kernel_selector;
    using kernel_params_t = kernel_selector::eye_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::eye_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<eye_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<eye>();
        auto params = get_default_params<kernel_selector::eye_params>(impl_param);

        params.diagonal_index = primitive->shift;

        return params;
    }
};

namespace detail {

attach_eye_impl::attach_eye_impl() {
    const std::vector<data_types> types{data_types::f16,
                                        data_types::f32,
                                        data_types::i8,
                                        data_types::u8,
                                        data_types::i32,
                                        data_types::i64};
    const std::vector<format::type> formats{
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
    };
    implementation_map<eye>::add(impl_types::ocl, typed_primitive_impl_ocl<eye>::create<eye_impl>, types, formats);
}

}  // namespace detail

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::eye_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::eye)
