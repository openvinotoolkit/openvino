// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eddo/experimental_detectron_detection_output_kernel_ref.h"
#include "eddo/experimental_detectron_detection_output_kernel_selector.h"
#include "experimental_detectron_detection_output_inst.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {
struct experimental_detectron_detection_output_impl
    : public typed_primitive_impl_ocl<experimental_detectron_detection_output> {
    using parent = typed_primitive_impl_ocl<experimental_detectron_detection_output>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<experimental_detectron_detection_output_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(typed_primitive_inst<experimental_detectron_detection_output>& instance,
                                        int32_t unused) const override {
        kernel_arguments_data args = parent::get_arguments(instance, unused);
        args.inputs.push_back(instance.output_classes_memory());
        args.inputs.push_back(instance.output_scores_memory());

        return args;
    }

public:
    static primitive_impl* create(const experimental_detectron_detection_output_node& arg) {
        auto params = get_default_params<kernel_selector::experimental_detectron_detection_output_params>(arg);
        auto optional_params =
            get_default_optional_params<kernel_selector::experimental_detectron_detection_output_optional_params>(
                arg.get_program());

        const auto& primitive = arg.get_primitive();

        params.score_threshold = primitive->score_threshold;
        params.nms_threshold = primitive->nms_threshold;
        params.max_delta_log_wh = primitive->max_delta_log_wh;
        params.num_classes = primitive->num_classes;
        params.post_nms_count = primitive->post_nms_count;
        params.max_detections_per_image = primitive->max_detections_per_image;
        params.class_agnostic_box_regression = primitive->class_agnostic_box_regression;
        params.deltas_weights = primitive->deltas_weights;

        params.inputs.push_back(convert_data_tensor(arg.deltas().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.scores().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.image_size_info().get_output_layout()));

        params.inputs.push_back(convert_data_tensor(arg.output_classes_node().get_output_layout()));
        params.inputs.push_back(convert_data_tensor(arg.output_scores_node().get_output_layout()));

        const auto& kernel_selector =
            kernel_selector::experimental_detectron_detection_output_kernel_selector::Instance();
        const auto best_kernels = kernel_selector.GetBestKernels(params, optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "best_kernels.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        return new experimental_detectron_detection_output_impl(arg, best_kernels[0]);
    }
};

namespace detail {
attach_experimental_detectron_detection_output_impl::attach_experimental_detectron_detection_output_impl() {
    implementation_map<experimental_detectron_detection_output>::add(
        impl_types::ocl,
        experimental_detectron_detection_output_impl::create,
        {std::make_tuple(data_types::f16, format::bfyx), std::make_tuple(data_types::f32, format::bfyx)});
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
