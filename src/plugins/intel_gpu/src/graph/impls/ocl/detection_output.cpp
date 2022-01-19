// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "detection_output/detection_output_kernel_selector.h"
#include "detection_output/detection_output_kernel_ref.h"
#include <vector>

namespace cldnn {
namespace ocl {

struct detection_output_impl : typed_primitive_impl_ocl<detection_output> {
    using parent = typed_primitive_impl_ocl<detection_output>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<detection_output_impl>(*this);
    }

private:
    static void set_detection_output_specific_params(kernel_selector::detection_output_params::DedicatedParams& detectOutParams,
                                                     const detection_output_node& arg) {
        auto primitive = arg.get_primitive();
        detectOutParams.keep_top_k = primitive->keep_top_k;
        detectOutParams.num_classes = primitive->num_classes;
        detectOutParams.top_k = primitive->top_k;
        detectOutParams.background_label_id = primitive->background_label_id;
        detectOutParams.code_type = (int32_t)primitive->code_type;
        detectOutParams.share_location = primitive->share_location;
        detectOutParams.variance_encoded_in_target = primitive->variance_encoded_in_target;
        detectOutParams.nms_threshold = primitive->nms_threshold;
        detectOutParams.eta = primitive->eta;
        detectOutParams.confidence_threshold = primitive->confidence_threshold;
        detectOutParams.prior_coordinates_offset = primitive->prior_coordinates_offset;
        detectOutParams.prior_info_size = primitive->prior_info_size;
        detectOutParams.prior_is_normalized = primitive->prior_is_normalized;
        detectOutParams.input_width = primitive->input_width;
        detectOutParams.input_heigh = primitive->input_height;
        detectOutParams.decrease_label_id = primitive->decrease_label_id;
        detectOutParams.clip_before_nms = primitive->clip_before_nms;
        detectOutParams.clip_after_nms = primitive->clip_after_nms;
        detectOutParams.conf_size_x = arg.confidence().get_output_layout().get_buffer_size().spatial[0];
        detectOutParams.conf_size_y = arg.confidence().get_output_layout().get_buffer_size().spatial[1];
        detectOutParams.conf_padding_x = arg.confidence().get_output_layout().data_padding.lower_size().spatial[0];
        detectOutParams.conf_padding_y = arg.confidence().get_output_layout().data_padding.lower_size().spatial[1];
    }

public:
    static primitive_impl* create(const detection_output_node& arg) {
        auto detect_out_params = get_default_params<kernel_selector::detection_output_params>(arg);
        auto detect_out_optional_params =
            get_default_optional_params<kernel_selector::detection_output_optional_params>(arg.get_program());

        detect_out_params.inputs.push_back(convert_data_tensor(arg.confidence().get_output_layout()));
        detect_out_params.inputs.push_back(convert_data_tensor(arg.prior_box().get_output_layout()));
        set_detection_output_specific_params(detect_out_params.detectOutParams, arg);

        auto& kernel_selector = kernel_selector::detection_output_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(detect_out_params, detect_out_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto detection_output = new detection_output_impl(arg, best_kernels[0]);

        return detection_output;
    }
};

namespace detail {

attach_detection_output_impl::attach_detection_output_impl() {
    implementation_map<detection_output>::add(impl_types::ocl, detection_output_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx)
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
