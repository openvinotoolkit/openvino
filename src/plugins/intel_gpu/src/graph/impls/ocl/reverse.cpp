// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "reverse_inst.h"
#include "reverse/reverse_kernel_ref.h"
#include "reverse/reverse_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct reverse_impl : typed_primitive_impl_ocl<reverse> {
    using parent = typed_primitive_impl_ocl<reverse>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reverse_kernel_selector;
    using kernel_params_t = kernel_selector::reverse_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::reverse_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<reverse_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<reverse>();
        auto params = get_default_params<kernel_selector::reverse_params>(impl_param);

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.reverseMode = primitive->mode == reverse_mode::index ? kernel_selector::reverse_mode::index
                                                                    : kernel_selector::reverse_mode::mask;
        return params;
    }
};

namespace detail {

attach_reverse_impl::attach_reverse_impl() {
    static const auto types =
        {data_types::f16, data_types::f32, data_types::i8, data_types::u8, data_types::i32, data_types::i64};
    static const auto formats = {
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

        format::bfwzyx,
    };
    std::set<std::tuple<data_types, format::type>> keys;
    for (const auto t : types) {
        for (const auto f : formats) {
            keys.emplace(t, f);
        }
    }
    implementation_map<reverse>::add(impl_types::ocl, typed_primitive_impl_ocl<reverse>::create<reverse_impl>, keys);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reverse_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::reverse)
