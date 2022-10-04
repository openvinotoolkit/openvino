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
        const auto& kernel_selector = kernel_selector::prior_box_kernel_selector::Instance();
        const auto& primitive = arg.get_primitive();

        const auto width = primitive->output_size.spatial[0];
        const auto height = primitive->output_size.spatial[1];
        const auto image_width = primitive->img_size.spatial[0];
        const auto image_height = primitive->img_size.spatial[1];

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
        const auto output_shape = impl_param.output_layout.get_shape();
        params.num_priors_4 = output_shape[1] / (params.width * params.height);

        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
        const auto best_kernels = kernel_selector.GetBestKernels(params, kernel_selector::prior_box_optional_params());
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");
        return new prior_box_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_prior_box_impl::attach_prior_box_impl() {
    auto types = {data_types::i32, data_types::i64};
    auto formats = {format::bfyx,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv16,
                    format::bs_fs_yx_bsv32_fsv32};
    implementation_map<prior_box>::add(impl_types::ocl, prior_box_impl::create, types, formats);
}
}  // namespace detail

}  // namespace ocl
}  // namespace cldnn
