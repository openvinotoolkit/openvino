// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample/grid_sample_kernel_ref.hpp"
#include "grid_sample/grid_sample_kernel_selector.hpp"
#include "grid_sample_inst.hpp"
#include "impls/implementation_map.hpp"
#include "primitive_base.hpp"

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

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<grid_sample_impl>(*this);
    }

    static primitive_impl* create(const grid_sample_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::grid_sample_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::grid_sample_optional_params>(arg.get_program());

        const auto grid_layout = impl_param.get_input_layout(1);
        params.inputs.push_back(convert_data_tensor(grid_layout));

        const auto primitive = impl_param.typed_desc<grid_sample>();
        params.align_corners = primitive->attributes.align_corners;
        params.interpolation_mode = from(primitive->attributes.mode);
        params.padding_mode = from(primitive->attributes.padding_mode);

        const auto& kernel_selector = kernel_selector::grid_sample_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new grid_sample_impl(arg, best_kernels.front());
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

    implementation_map<grid_sample>::add(impl_types::ocl, grid_sample_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
