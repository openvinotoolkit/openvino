// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <experimental_detectron_prior_grid_generator_inst.h>
#include "primitive_base.hpp"
#include <impls/implementation_map.hpp>
#include <kernel_selector_helper.h>
#include <experimental_detectron_prior_grid_generator/experimental_detectron_prior_grid_generator_kernel_selector.h>
#include <experimental_detectron_prior_grid_generator/experimental_detectron_prior_grid_generator_kernel_ref.h>
#include <intel_gpu/runtime/error_handler.hpp>

namespace cldnn {
namespace ocl {

struct experimental_detectron_prior_grid_generator_impl : typed_primitive_impl_ocl<experimental_detectron_prior_grid_generator> {
    using typed_primitive_impl_ocl::typed_primitive_impl_ocl;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_prior_grid_generator_impl>(*this);
    }

    static primitive_impl* create(const experimental_detectron_prior_grid_generator_node& arg) {
        auto params = get_default_params<kernel_selector::experimental_detectron_prior_grid_generator_params>(arg);
        auto primPtr = arg.get_primitive();
        auto &prim = *primPtr;

        params.flatten = prim.flatten;
        params.layer_height = prim.h ? prim.h : prim.featmap_height;
        params.layer_width = prim.w ? prim.w : prim.featmap_width;
        params.number_of_priors = prim.number_of_priors;
        params.step_x = prim.stride_x ? prim.stride_x : static_cast<float>(prim.image_width) / params.layer_width;
        params.step_y = prim.stride_y ? prim.stride_y : static_cast<float>(prim.image_height) / params.layer_height;

        auto optional_params =
            get_default_optional_params<kernel_selector::experimental_detectron_prior_grid_generator_optional_params>(arg.get_program());

        auto& kernel_selector = kernel_selector::experimental_detectron_prior_grid_generator_instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new experimental_detectron_prior_grid_generator_impl{arg, best_kernels.front()};
    }
};

namespace detail {

attach_experimental_detectron_prior_grid_generator_impl::attach_experimental_detectron_prior_grid_generator_impl() {
    implementation_map<experimental_detectron_prior_grid_generator>::add(
        impl_types::ocl,
        experimental_detectron_prior_grid_generator_impl::create,
        {
            std::make_tuple(data_types::f16, format::bfyx),
            std::make_tuple(data_types::f32, format::bfyx),
        });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

