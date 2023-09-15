// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "pyramid_roi_align_inst.h"
#include "pyramid_roi_align/pyramid_roi_align_kernel_selector.h"
#include "pyramid_roi_align/pyramid_roi_align_kernel_base.h"

#include <cmath>

namespace cldnn {
namespace ocl {

struct pyramid_roi_align_impl : typed_primitive_impl_ocl<pyramid_roi_align> {
    using parent = typed_primitive_impl_ocl<pyramid_roi_align>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::PyramidROIAlign_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::PyramidROIAlign_params, kernel_selector::PyramidROIAlign_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::pyramid_roi_align_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<pyramid_roi_align_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<pyramid_roi_align>();
        auto params = get_default_params<kernel_selector::PyramidROIAlign_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::PyramidROIAlign_optional_params>(impl_param.get_program());

        const auto P2_idx = 1;
        const auto P3_idx = 2;
        const auto P4_idx = 3;
        const auto P5_idx = 4;
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(P2_idx)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(P3_idx)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(P4_idx)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(P5_idx)));

        params.sampling_ratio_x = primitive->sampling_ratio;
        params.sampling_ratio_y = primitive->sampling_ratio;

        auto first_layer_scale = primitive->pyramid_scales[0];
        auto image_size_x = impl_param.get_input_layout(P2_idx).spatial(0) * first_layer_scale;
        auto image_size_y = impl_param.get_input_layout(P2_idx).spatial(1) * first_layer_scale;
        params.image_size_x = image_size_x;
        params.image_size_y = image_size_y;

        params.pyramid_starting_level = primitive->pyramid_starting_level;

        return {params, optional_params};
    }
};

namespace detail {

attach_pyramid_roi_align_impl::attach_pyramid_roi_align_impl() {
    implementation_map<pyramid_roi_align>::add(impl_types::ocl, typed_primitive_impl_ocl<pyramid_roi_align>::create<pyramid_roi_align_impl>, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::f16, format::byxf),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::pyramid_roi_align_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::pyramid_roi_align)
