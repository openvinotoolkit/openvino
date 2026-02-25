// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "experimental_detectron_prior_grid_generator_inst.h"
#include "ed_pgg/prior_grid_generator_kernel_ref.h"
#include "ed_pgg/prior_grid_generator_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct experimental_detectron_prior_grid_generator_impl
    : typed_primitive_impl_ocl<experimental_detectron_prior_grid_generator> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_prior_grid_generator>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::experimental_detectron_prior_grid_generator_kernel_selector;
    using kernel_params_t = kernel_selector::experimental_detectron_prior_grid_generator_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::experimental_detectron_prior_grid_generator_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<experimental_detectron_prior_grid_generator_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<experimental_detectron_prior_grid_generator>();
        auto params = get_default_params<kernel_selector::experimental_detectron_prior_grid_generator_params>(impl_param);

        params.flatten = primitive->flatten;
        params.layer_height = primitive->h ? primitive->h : primitive->featmap_height;
        params.layer_width = primitive->w ? primitive->w : primitive->featmap_width;
        params.step_x = primitive->stride_x ? primitive->stride_x : static_cast<float>(primitive->image_width) / params.layer_width;
        params.step_y = primitive->stride_y ? primitive->stride_y : static_cast<float>(primitive->image_height) / params.layer_height;

        return params;
    }
};

namespace detail {

attach_experimental_detectron_prior_grid_generator_impl::attach_experimental_detectron_prior_grid_generator_impl() {
    implementation_map<experimental_detectron_prior_grid_generator>::add(
        impl_types::ocl,
        typed_primitive_impl_ocl<experimental_detectron_prior_grid_generator>::create<experimental_detectron_prior_grid_generator_impl>,
        {
            std::make_tuple(data_types::f16, format::bfyx),
            std::make_tuple(data_types::f32, format::bfyx),
        });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::experimental_detectron_prior_grid_generator_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::experimental_detectron_prior_grid_generator)
