// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "prior_box_inst.h"
#include "prior_box/prior_box_kernel_ref.h"
#include "prior_box/prior_box_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct prior_box_impl : typed_primitive_impl_ocl<prior_box> {
    using parent = typed_primitive_impl_ocl<prior_box>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::prior_box_kernel_selector;
    using kernel_params_t = kernel_selector::prior_box_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::prior_box_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<prior_box_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<prior_box>();
        auto params = get_default_params<kernel_selector::prior_box_params>(impl_param);

        auto width = primitive->output_size.spatial[0];
        auto height = primitive->output_size.spatial[1];
        auto image_width = primitive->img_size.spatial[0];
        auto image_height = primitive->img_size.spatial[1];

        if (width == 0 || height == 0 || image_width == 0 || image_height == 0) {
            width = impl_param.output_size[0];
            height = impl_param.output_size[1];
            image_width = impl_param.img_size[0];
            image_height = impl_param.img_size[1];
        }
        params.min_size = primitive->min_sizes;
        params.max_size = primitive->max_sizes;
        params.density = primitive->density;
        params.fixed_ratio = primitive->fixed_ratio;
        params.fixed_size = primitive->fixed_size;
        params.clip = primitive->clip;
        params.flip = primitive->flip;
        params.scale_all_sizes = primitive->scale_all_sizes;
        params.step = primitive->step;
        float step = params.step;
        if (!params.scale_all_sizes) {
            // mxnet-like PriorBox
            if (step == -1) {
                step = 1.f * image_height / height;
            } else {
                step *= image_height;
            }
            for (auto& size : params.min_size) {
                size *= image_height;
            }
        }
        params.offset = primitive->offset;
        params.min_max_aspect_ratios_order = primitive->min_max_aspect_ratios_order;
        params.aspect_ratio = primitive->aspect_ratios;
        params.variance = primitive->variance;
        params.reverse_image_width = 1.0f / image_width;
        params.reverse_image_height = 1.0f / image_height;
        params.width = width;
        params.height = height;
        if (step == 0) {
            params.step_x = image_width / width;
            params.step_y = image_height / height;
        } else {
            params.step_x = step;
            params.step_y = step;
        }
        params.widths = primitive->widths;
        params.heights = primitive->heights;
        const auto output_shape = impl_param.get_output_layout().get_shape();
        params.num_priors_4 = static_cast<uint32_t>(output_shape[1] / (params.width * params.height));

        params.is_clustered = primitive->is_clustered();

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        return params;
    }
};

namespace detail {

attach_prior_box_impl::attach_prior_box_impl() {
    auto types = {data_types::i32, data_types::i64, data_types::f32, data_types::f16};
    auto formats = {format::bfyx,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv16,
                    format::bs_fs_yx_bsv32_fsv32};
    implementation_map<prior_box>::add(impl_types::ocl, typed_primitive_impl_ocl<prior_box>::create<prior_box_impl>, types, formats);
}
}  // namespace detail

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::prior_box_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::prior_box)
