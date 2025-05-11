// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "grid_sample_inst.hpp"
#include "grid_sample/grid_sample_kernel_ref.hpp"
#include "grid_sample/grid_sample_kernel_selector.hpp"

namespace cldnn {
namespace ocl {

namespace {

kernel_selector::grid_sample_params::InterpolationMode from(GridSampleOp::InterpolationMode interpolation_mode) {
    switch (interpolation_mode) {
    default:
    case GridSampleOp::InterpolationMode::BILINEAR:
        return kernel_selector::grid_sample_params::InterpolationMode::BILINEAR;
    case GridSampleOp::InterpolationMode::BICUBIC:
        return kernel_selector::grid_sample_params::InterpolationMode::BICUBIC;
    case GridSampleOp::InterpolationMode::NEAREST:
        return kernel_selector::grid_sample_params::InterpolationMode::NEAREST;
    }
}

kernel_selector::grid_sample_params::PaddingMode from(GridSampleOp::PaddingMode padding_mode) {
    switch (padding_mode) {
    default:
    case GridSampleOp::PaddingMode::ZEROS:
        return kernel_selector::grid_sample_params::PaddingMode::ZEROS;
    case GridSampleOp::PaddingMode::BORDER:
        return kernel_selector::grid_sample_params::PaddingMode::BORDER;
    case GridSampleOp::PaddingMode::REFLECTION:
        return kernel_selector::grid_sample_params::PaddingMode::REFLECTION;
    }
}

}  // namespace

struct grid_sample_impl : public typed_primitive_impl_ocl<grid_sample> {
    using parent = typed_primitive_impl_ocl<grid_sample>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::grid_sample_kernel_selector;
    using kernel_params_t = kernel_selector::grid_sample_params;


    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::grid_sample_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<grid_sample_impl, kernel_params_t>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<grid_sample>();
        auto params = get_default_params<kernel_selector::grid_sample_params>(impl_param);

        const auto grid_layout = impl_param.get_input_layout(1);
        params.inputs.push_back(convert_data_tensor(grid_layout));

        params.align_corners = primitive->attributes.align_corners;
        params.interpolation_mode = from(primitive->attributes.mode);
        params.padding_mode = from(primitive->attributes.padding_mode);

        return params;
    }
};

namespace detail {

attach_grid_sample_impl::attach_grid_sample_impl() {
    auto types = {data_types::u8, data_types::i8, data_types::f16, data_types::f32, data_types::i32, data_types::i64};

    auto formats = {format::bfyx,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv32,
                    format::bs_fs_yx_bsv32_fsv16};

    implementation_map<grid_sample>::add(impl_types::ocl, typed_primitive_impl_ocl<grid_sample>::create<grid_sample_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::grid_sample_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::grid_sample)
