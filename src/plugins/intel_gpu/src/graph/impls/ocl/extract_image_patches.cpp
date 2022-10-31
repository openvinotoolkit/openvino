// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extract_image_patches_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"

#include "extract_image_patches/extract_image_patches_kernel_selector.h"
#include "extract_image_patches/extract_image_patches_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct extract_image_patches_impl : typed_primitive_impl_ocl<extract_image_patches> {
    using parent = typed_primitive_impl_ocl<extract_image_patches>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<extract_image_patches_impl>(*this);
    }

public:
    static std::unique_ptr<primitive_impl> create(const extract_image_patches_node& arg, const kernel_impl_params& impl_param) {
        const auto& prim = arg.get_primitive();
        auto params = get_default_params<kernel_selector::extract_image_patches_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::extract_image_patches_optional_params>(arg.get_program());

        params.sizes = prim->sizes;
        params.strides = prim->strides;
        params.rates = prim->rates;
        params.auto_pad = prim->auto_pad;

        auto& kernel_selector = kernel_selector::extract_image_patches_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(params, optional_params);

        return make_unique<extract_image_patches_impl>(arg, best_kernel);
    }
};

namespace detail {

attach_extract_image_patches_impl::attach_extract_image_patches_impl() {
    implementation_map<extract_image_patches>::add(impl_types::ocl, extract_image_patches_impl::create, {
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

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::extract_image_patches_impl, cldnn::object_type::EXTRACT_IMAGE_PATCHES_IMPL)