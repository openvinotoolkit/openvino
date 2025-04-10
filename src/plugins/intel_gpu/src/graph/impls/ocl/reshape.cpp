// Copyright (C) 2018-2025 Intel Corporation
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
    using kernel_params_t = kernel_selector::reshape_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::reshape_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<reshape_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        return get_default_params<kernel_selector::reshape_params>(impl_param);
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override { }
};

namespace detail {

attach_reshape_impl::attach_reshape_impl() {
    implementation_map<reshape>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<reshape>::create<reshape_impl>, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reshape_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::reshape)
