// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "extract_image_patches_inst.h"
#include "extract_image_patches/extract_image_patches_kernel_selector.h"
#include "extract_image_patches/extract_image_patches_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct extract_image_patches_impl : typed_primitive_impl_ocl<extract_image_patches> {
    using parent = typed_primitive_impl_ocl<extract_image_patches>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::extract_image_patches_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::extract_image_patches_params, kernel_selector::extract_image_patches_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::extract_image_patches_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<extract_image_patches_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<extract_image_patches>();
        auto params = get_default_params<kernel_selector::extract_image_patches_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::extract_image_patches_optional_params>(impl_param.get_program());

        params.sizes = primitive->sizes;
        params.strides = primitive->strides;
        params.rates = primitive->rates;
        params.auto_pad = primitive->auto_pad;

        return {params, optional_params};
    }
};

namespace detail {

attach_extract_image_patches_impl::attach_extract_image_patches_impl() {
    implementation_map<extract_image_patches>::add(impl_types::ocl, typed_primitive_impl_ocl<extract_image_patches>::create<extract_image_patches_impl>, {
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::extract_image_patches_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::extract_image_patches)
