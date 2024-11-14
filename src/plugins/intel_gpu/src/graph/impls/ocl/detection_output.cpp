// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "detection_output.hpp"
#include "detection_output_inst.h"
#include "detection_output/detection_output_kernel_selector.h"
#include "detection_output/detection_output_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct detection_output_impl : typed_primitive_impl_ocl<detection_output> {
    using parent = typed_primitive_impl_ocl<detection_output>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::detection_output_kernel_selector;
    using kernel_params_t = kernel_selector::detection_output_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::detection_output_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<detection_output_impl>(*this);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<detection_output>();
        auto params = get_default_params<kernel_selector::detection_output_params>(impl_param);

        const auto confidence_idx = 1;
        const auto prior_box_idx = 2;
        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[confidence_idx]));
        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[prior_box_idx]));

        auto confidence_layout = impl_param.get_input_layout(1);
        auto& detectOutParams = params.detectOutParams;
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
        detectOutParams.conf_size_x = confidence_layout.get_padded_dims()[2];
        detectOutParams.conf_size_y = confidence_layout.get_padded_dims()[3];
        detectOutParams.conf_padding_x = confidence_layout.data_padding._lower_size[2];
        detectOutParams.conf_padding_y = confidence_layout.data_padding._lower_size[3];

        return params;
    }
};

std::unique_ptr<primitive_impl> DetectionOutputImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<detection_output>());
    return typed_primitive_impl_ocl<detection_output>::create<detection_output_impl>(static_cast<const detection_output_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::detection_output_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::detection_output)
