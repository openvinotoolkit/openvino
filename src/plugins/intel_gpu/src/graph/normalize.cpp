// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(normalize)

layout normalize_inst::calc_output_layout(normalize_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for normalize_node!");
    auto input_node_layout = impl_param.get_non_padded_input_layout();
    auto output_type = input_node_layout.data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_output_element_type();
    } else if (input_node_layout.data_type == data_types::u8 || input_node_layout.data_type == data_types::i8) {
        output_type = data_types::f32;
    }

    return layout(output_type, input_node_layout.format, input_node_layout.get_tensor());
}

std::string normalize_inst::to_string(normalize_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto epsilon = desc->epsilon;
    auto norm_region = desc->across_spatial ? "across spatial" : "within spatial";
    auto& input = node.input();
    auto& scale_input = node.scale();

    std::stringstream primitive_description;

    json_composite normalize_info;
    normalize_info.add("input id", input.id());
    normalize_info.add("scale input id", scale_input.id());
    normalize_info.add("epsilon", epsilon);
    normalize_info.add("normalization region", norm_region);

    node_info->add("normalize info", normalize_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

normalize_inst::typed_primitive_inst(network& network, normalize_node const& node) : parent(network, node) {
    if (node.input().is_dynamic() || node.scale().is_dynamic())
        return;
    /// Scale f dimension should be 1 (if all channels have the same scale) or equal to input feature size (one scale per channel).
    auto scale_layout = node.scale().get_output_layout();
    auto scale_size = scale_layout.get_tensor();
    auto scale_feature_size = scale_size.feature[0];
    auto input_layout = node.get_input_layout();
    auto input_feature_size = input_layout.feature();

    if (scale_feature_size != 1) {
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Scale feature size",
                              scale_feature_size,
                              "input feature size",
                              input_feature_size,
                              "");
    }

    // All other dimensions should be 1
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Scale input size elements count",
                          (int32_t)scale_size.count(),
                          "scale feature size",
                          scale_feature_size,
                          "Dimensions mismatch of scale input in Normalize layer!");
}
}  // namespace cldnn
