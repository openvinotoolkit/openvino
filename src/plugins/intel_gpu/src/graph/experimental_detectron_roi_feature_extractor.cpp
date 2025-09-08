// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(experimental_detectron_roi_feature_extractor)

size_t experimental_detectron_roi_feature_extractor_inst::inputs_memory_count() const {
    return parent::inputs_memory_count() - 1;
}

memory::ptr experimental_detectron_roi_feature_extractor_inst::second_output_memory() const {
    if (desc()->num_outputs == 1) {
        return input_memory_ptr(parent::inputs_memory_count() - 1);
    } else {
        return output_memory_ptr(1);
    }
}

memory::ptr experimental_detectron_roi_feature_extractor_inst::rois_memory() const {
    return input_memory_ptr(0);
}

void experimental_detectron_roi_feature_extractor_inst::copy_rois_input_to_second_output() const {
    second_output_memory()->copy_from(get_network().get_stream(), *rois_memory());
}

template<typename ShapeType>
std::vector<layout> experimental_detectron_roi_feature_extractor_inst::calc_output_layouts(
        experimental_detectron_roi_feature_extractor_node const& /*node*/, const kernel_impl_params& impl_param) {
    layout rois_layout = impl_param.get_input_layout(0);
    layout data_layout = impl_param.get_input_layout(1);
    auto desc = impl_param.typed_desc<experimental_detectron_roi_feature_extractor>();
    auto num_rois = rois_layout.get_partial_shape()[0];
    auto num_channels = data_layout.get_partial_shape()[1];

    return {
        layout(ov::PartialShape{num_rois, num_channels, desc->output_dim, desc->output_dim}, data_layout.data_type, format::bfyx),
        layout(ov::PartialShape{num_rois, 4}, data_layout.data_type, format::bfyx)
    };
}

template std::vector<layout>
experimental_detectron_roi_feature_extractor_inst::calc_output_layouts<ov::PartialShape>(
        experimental_detectron_roi_feature_extractor_node const& node, const kernel_impl_params& impl_param);

layout experimental_detectron_roi_feature_extractor_inst::calc_output_layout(
    experimental_detectron_roi_feature_extractor_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for roi_pooling_node!");
    layout rois_layout = impl_param.get_input_layout(0);
    layout data_layout = impl_param.get_input_layout(1);
    int num_rois = rois_layout.batch();
    int num_channels = data_layout.feature();
    auto desc = impl_param.typed_desc<experimental_detectron_roi_feature_extractor>();

    return layout(data_layout.data_type, format::bfyx, {num_rois, num_channels, desc->output_dim, desc->output_dim});
}

std::string experimental_detectron_roi_feature_extractor_inst::to_string(experimental_detectron_roi_feature_extractor_node const& node) {
    auto desc = node.get_primitive();

    std::stringstream primitive_description;

    json_composite experimental_detectron_info;
    experimental_detectron_info.add("output_size", desc->output_dim);
    experimental_detectron_info.add("pooled_height", desc->pooled_height);
    experimental_detectron_info.add("pooled_width", desc->pooled_width);
    experimental_detectron_info.add("sampling_ratio", desc->sampling_ratio);
    for (std::size_t i = 0; i < desc->pyramid_scales.size(); i++) {
        experimental_detectron_info.add("pyramid_scales[" + std::to_string(i) + "]", desc->pyramid_scales[i]);
    }
    experimental_detectron_info.add("aligned", (desc->aligned ? "true" : "false"));

    auto node_info = node.desc_to_json();
    node_info->add("experimental_detectron_info", experimental_detectron_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
