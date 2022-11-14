// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "reshape/reshape_kernel_ref.h"
#include "reshape/reshape_kernel_selector.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct reshape_impl : public typed_primitive_impl_ocl<reshape> {
    using parent = typed_primitive_impl_ocl<reshape>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::reshape_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::reshape_params, kernel_selector::reshape_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<reshape_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::reshape_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::reshape_optional_params>(impl_param.get_program());

        return {params, optional_params};
    }
};

namespace detail {

attach_reshape_impl::attach_reshape_impl() {
    implementation_map<reshape>::add(impl_types::ocl, typed_primitive_impl_ocl<reshape>::create<reshape_impl>, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::reshape_impl, cldnn::object_type::RESHAPE_IMPL)
