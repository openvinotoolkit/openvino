// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "roi_align_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "roi_align/roi_align_kernel_selector.h"
#include "roi_align/roi_align_kernel_ref.h"

namespace cldnn {
namespace ocl {

namespace {
kernel_selector::pool_type from(roi_align::PoolingMode mode) {
    switch (mode) {
        case roi_align::PoolingMode::Max:
            return kernel_selector::pool_type::MAX;
        default:
        case roi_align::PoolingMode::Avg:
            return kernel_selector::pool_type::AVG;
    }
}
}  // namespace

struct roi_align_impl : typed_primitive_impl_ocl<roi_align> {
    using parent = typed_primitive_impl_ocl<roi_align>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<roi_align_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<roi_align>& instance, int32_t) const override {
        kernel_arguments_data args;
        args.inputs = { instance.input_memory_ptr(), instance.rois_memory(), instance.batches_memory() };
        args.outputs = { instance.output_memory_ptr() };

        return args;
    }

public:
    static primitive_impl* create(const roi_align_node& arg) {
        const auto& input_layout = arg.input().get_output_layout();
        const auto& output_layout = arg.get_output_layout();
        const auto& rois_layout = arg.input(1).get_output_layout();
        const auto& batches_layout = arg.input(2).get_output_layout();
        const auto& primitive = arg.get_primitive();

        const auto padding_filling_value = output_layout.data_padding.filling_value();

        CLDNN_ERROR_NOT_EQUAL(arg.id(),
                              "roi_align padding filling value",
                              padding_filling_value,
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in roi_align.");
        CLDNN_ERROR_NOT_PROPER_FORMAT(arg.id(),
                                      "Input_layout.format",
                                      input_layout.format.value,
                                      "output_layout.format",
                                      output_layout.format);

        auto roi_align_params = get_default_params<kernel_selector::roi_align_params>(arg);
        auto roi_align_optional_params =
            get_default_optional_params<kernel_selector::roi_align_optional_params>(arg.get_program());

        const auto roi_bfyx = convert_data_tensor(rois_layout);
        roi_align_params.inputs.push_back(roi_bfyx.FlattenFeatureAndSpatials());
        roi_align_params.inputs.push_back(convert_data_tensor(batches_layout));
        roi_align_params.mode = from(primitive->mode);
        roi_align_params.sampling_ratio = primitive->sampling_ratio;
        roi_align_params.spatial_scale = primitive->spatial_scale;

        auto& kernel_selector = kernel_selector::roi_align_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(roi_align_params, roi_align_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto roi_align = new roi_align_impl(arg, best_kernels[0]);

        return roi_align;
    }
};

namespace detail {

attach_roi_align_impl::attach_roi_align_impl() {
    implementation_map<roi_align>::add(impl_types::ocl, roi_align_impl::create,
                                       {
                                           std::make_tuple(data_types::f16, format::bfyx),
                                           std::make_tuple(data_types::f32, format::bfyx),
                                       });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
