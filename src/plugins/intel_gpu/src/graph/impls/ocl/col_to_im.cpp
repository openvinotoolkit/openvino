// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "col_to_im_inst.h"
#include "col_to_im/col_to_im_kernel_selector.h"
#include "col_to_im/col_to_im_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct col_to_im_impl : typed_primitive_impl_ocl<col_to_im> {
    using parent = typed_primitive_impl_ocl<col_to_im>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::col_to_im_kernel_selector;
    using kernel_params_t = kernel_selector::col_to_im_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::col_to_im_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<col_to_im_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<col_to_im>();
        auto params = get_default_params<kernel_selector::col_to_im_params>(impl_param);

        return params;
    }
};

namespace detail {

attach_col_to_im_impl::attach_col_to_im_impl() {
    std::vector<data_types> dt = {
        data_types::f16,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
    };
    implementation_map<col_to_im>::add(impl_types::ocl, typed_primitive_impl_ocl<col_to_im>::create<col_to_im_impl>, dt, fmt);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::col_to_im_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::col_to_im)
