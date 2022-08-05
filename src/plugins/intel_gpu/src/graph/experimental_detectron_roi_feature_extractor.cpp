// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_roi_feature_extractor_inst.hpp"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id experimental_detectron_roi_feature_extractor::type_id() {
    static primitive_type_base<experimental_detectron_roi_feature_extractor> instance;
    return &instance;
}

size_t experimental_detectron_roi_feature_extractor_inst::inputs_memory_count() const {
    return parent::inputs_memory_count() - 1;
}

memory::ptr experimental_detectron_roi_feature_extractor_inst::second_output_memory() const {
    return input_memory_ptr(parent::inputs_memory_count() - 1);
}

memory::ptr experimental_detectron_roi_feature_extractor_inst::rois_memory() const {
    return input_memory_ptr(0);
}

void experimental_detectron_roi_feature_extractor_inst::copy_rois_input_to_second_output() const {
    second_output_memory()->copy_from(get_network().get_stream(), *rois_memory());
}

layout experimental_detectron_roi_feature_extractor_inst::calc_output_layout(experimental_detectron_roi_feature_extractor_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for roi_pooling_node!");
    layout rois_layout = node.input(0).get_output_layout();
    layout data_layout = node.input(1).get_output_layout();
    int num_rois = rois_layout.batch();
    int num_channels = data_layout.feature();
    auto desc = node.get_primitive();

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
