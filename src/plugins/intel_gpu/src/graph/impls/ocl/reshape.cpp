// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "reshape_inst.h"
#include "reshape/reshape_kernel_ref.h"
#include "reshape/reshape_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct reshape_impl : public typed_primitive_impl_ocl<reshape> {
    using parent = typed_primitive_impl_ocl<reshape>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reshape_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::reshape_params, kernel_selector::reshape_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::reshape_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reshape_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::reshape_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::reshape_optional_params>(impl_param.get_program());

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override { }
};

namespace detail {

attach_reshape_impl::attach_reshape_impl() {
    implementation_map<reshape>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<reshape>::create<reshape_impl>, {});

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
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx
    };

    implementation_map<reshape>::add(impl_types::ocl,
                                     shape_types::dynamic_shape,
                                     typed_primitive_impl_ocl<reshape>::create<reshape_impl>,
                                     dyn_types,
                                     dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reshape_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::reshape)
