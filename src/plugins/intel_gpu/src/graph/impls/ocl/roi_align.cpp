// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "roi_align_inst.h"
#include "roi_align/roi_align_kernel_ref.h"
#include "roi_align/roi_align_kernel_selector.h"

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::pool_type from(roi_align::PoolingMode mode) {
    switch (mode) {
    case roi_align::PoolingMode::max:
        return kernel_selector::pool_type::MAX;
    default:
    case roi_align::PoolingMode::avg:
        return kernel_selector::pool_type::AVG;
    }
}

kernel_selector::roi_aligned_mode from(roi_align::AlignedMode mode) {
    switch (mode) {
    case roi_align::AlignedMode::half_pixel_for_nn:
        return kernel_selector::roi_aligned_mode::HALF_PIXEL_FOR_NN;
    case roi_align::AlignedMode::half_pixel:
        return kernel_selector::roi_aligned_mode::HALF_PIXEL;
    default:
    case roi_align::AlignedMode::asymmetric:
        return kernel_selector::roi_aligned_mode::ASYMMETRIC;
    }
}
}  // namespace

struct roi_align_impl : typed_primitive_impl_ocl<roi_align> {
    using parent = typed_primitive_impl_ocl<roi_align>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::roi_align_kernel_selector;
    using kernel_params_t = kernel_selector::roi_align_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::roi_align_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<roi_align_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<roi_align>& instance) const override {
        kernel_arguments_data args;
        args.inputs = {instance.input_memory_ptr(), instance.rois_memory(), instance.batches_memory()};
        args.outputs = {instance.output_memory_ptr()};

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<roi_align>();
        const auto& rois_layout = impl_param.get_input_layout(1);
        const auto& batches_layout = impl_param.get_input_layout(2);

        auto params = get_default_params<kernel_selector::roi_align_params>(impl_param);

        params.inputs.push_back(convert_data_tensor(rois_layout));
        params.inputs.push_back(convert_data_tensor(batches_layout));
        params.pooling_mode = from(primitive->pooling_mode);
        params.aligned_mode = from(primitive->aligned_mode);
        params.sampling_ratio = primitive->sampling_ratio;
        params.spatial_scale = primitive->spatial_scale;
        params.rotated_mode = primitive->roi_mode == roi_align::ROIMode::rotated;
        params.clockwise = primitive->clockwise;

        return params;
    }
};

namespace detail {

attach_roi_align_impl::attach_roi_align_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8, data_types::i32};

    auto formats = {format::bfyx,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv16,
                    format::bs_fs_yx_bsv32_fsv32};

    implementation_map<roi_align>::add(impl_types::ocl, typed_primitive_impl_ocl<roi_align>::create<roi_align_impl>, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::roi_align_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::roi_align)
