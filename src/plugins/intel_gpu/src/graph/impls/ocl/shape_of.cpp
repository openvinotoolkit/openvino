// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "shape_of/shape_of_kernel_selector.h"
#include "shape_of/shape_of_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct shape_of_impl : typed_primitive_impl_ocl<shape_of> {
    using parent = typed_primitive_impl_ocl<shape_of>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::shape_of_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::shape_of_params, kernel_selector::shape_of_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<shape_of_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::shape_of_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::shape_of_optional_params>(impl_param.get_program());

        auto input_layout = impl_param.get_input_layout(0);
        params.input_rank = input_layout.get_rank();
        params.input_dims = input_layout.get_dims();

        return {params, optional_params};
    }
};

namespace detail {

attach_shape_of_impl::attach_shape_of_impl() {
    implementation_map<shape_of>::add(impl_types::ocl, typed_primitive_impl_ocl<shape_of>::create<shape_of_impl>, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::shape_of_impl, cldnn::object_type::SHAPE_OF_IMPL)
