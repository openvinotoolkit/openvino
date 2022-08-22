// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <prior_box/prior_box_kernel_ref.h>
#include <prior_box/prior_box_kernel_selector.h>
#include <prior_box_inst.h>

#include <impls/implementation_map.hpp>
#include <vector>

#include "intel_gpu/runtime/error_handler.hpp"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct prior_box_impl : typed_primitive_impl_ocl<prior_box> {
    using parent = typed_primitive_impl_ocl<prior_box>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<prior_box_impl>(*this);
    }

    static primitive_impl* create(const prior_box_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::prior_box_params>(impl_param);
        auto& kernel_selector = kernel_selector::prior_box_kernel_selector::Instance();
        const auto& primitive = arg.get_primitive();

        params.min_size = primitive->attributes.min_size;
        params.max_size = primitive->attributes.max_size;
        params.density = primitive->attributes.density;
        params.fixed_ratio = primitive->attributes.fixed_ratio;
        params.fixed_size = primitive->attributes.fixed_size;
        params.clip = primitive->attributes.clip;
        params.flip = primitive->attributes.flip;
        params.step = primitive->attributes.step;
        params.offset = primitive->attributes.offset;
        params.scale_all_sizes = primitive->attributes.scale_all_sizes;
        params.min_max_aspect_ratios_order = primitive->attributes.min_max_aspect_ratios_order;
        params.aspect_ratio = primitive->aspect_ratios;
        params.variance = primitive->variance;
        params.reverse_image_width = primitive->reverse_image_width;
        params.reverse_image_height = primitive->reverse_image_height;
        params.step_x = primitive->step_x;
        params.step_y = primitive->step_y;
        params.width = primitive->width;
        params.height = primitive->height;
        params.widths = primitive->attributes.widths;
        params.heights = primitive->attributes.heights;
        params.step_widths = primitive->attributes.step_widths;
        params.step_heights = primitive->attributes.step_heights;
        params.is_clustered = primitive->is_clustered();
        auto output_shape = impl_param.output_layout.get_shape();
        params.num_priors_4 = output_shape[1] / (params.width * params.height);

        params.inputs.pop_back();
        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[0]));
        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
        auto best_kernels = kernel_selector.GetBestKernels(params, kernel_selector::prior_box_optional_params());
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");
        return new prior_box_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_prior_box_impl::attach_prior_box_impl() {
    implementation_map<prior_box>::add(impl_types::ocl,
                                       prior_box_impl::create,
                                       {
                                           std::make_tuple(data_types::i32, format::bfyx),
                                           std::make_tuple(data_types::i32, format::bfzyx),
                                           std::make_tuple(data_types::i32, format::bfwzyx),
                                           std::make_tuple(data_types::i64, format::bfyx),
                                           std::make_tuple(data_types::i64, format::bfzyx),
                                           std::make_tuple(data_types::i64, format::bfwzyx),
                                       });
}
}  // namespace detail

}  // namespace ocl
}  // namespace cldnn
