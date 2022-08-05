// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "deconvolution_inst.h"
#include "primitive_type_base.h"
#include "sliding_window_utils.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

using namespace ov::runtime::intel_gpu;

namespace cldnn {
primitive_type_id deconvolution::type_id() {
    static primitive_type_base<deconvolution> instance;
    return &instance;
}

layout deconvolution_inst::calc_output_layout(deconvolution_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for deconvolution_node!");
    auto desc = node.get_primitive();

    auto input_layout = node.input().get_output_layout();
    auto weights_layout = node.weights(0).get_output_layout().convert_to_weights_layout(desc->grouped_weights_shape);

    auto data_type = input_layout.data_type;
    if ((input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8) && !node.has_fused_primitives()) {
        data_type = data_types::f32;
    }

    if (node.has_fused_primitives()) {
        data_type = node.get_fused_output_layout().data_type;
    }

    auto pad = desc->pad;
    auto strd = desc->stride;

    int32_t number_of_features = weights_layout.group() * weights_layout.ofm();

    if (desc->with_output_size) {
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User-defined output spatial X",
                                       desc->output_size.spatial[0],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User-defined output spatial Y",
                                       desc->output_size.spatial[1],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");
        CLDNN_ERROR_LESS_OR_EQUAL_THAN(node.id(),
                                       "User-defined output spatial Z",
                                       desc->output_size.spatial[2],
                                       "value 0",
                                       0,
                                       "User-defined size of output layout must be positive (>= 1)");

        tensor output_size(input_layout.batch(),
                           number_of_features,
                           desc->output_size.spatial[0],
                           desc->output_size.spatial[1],
                           desc->output_size.spatial[2]);
        return {data_type, input_layout.format, output_size};
    }

    int32_t off_factor = -2;
    size_t spatial_dims = input_layout.get_spatial_rank();
    CLDNN_ERROR_GREATER_THAN(node.id(),
                             "number of spatial dimensions",
                             spatial_dims,
                             "expected number of dimensions",
                             3,
                             "As for now, deconvolutions with more than 3 dimensions are not supported");

    int32_t x = off_factor * pad[pad.size() - 1] + (input_layout.spatial(0) - 1) * strd[strd.size() - 1] + weights_layout.spatial(0);
    int32_t y = 1;
    if (spatial_dims > 1) {
        y = off_factor * pad[pad.size() - 2] + (input_layout.spatial(1) - 1) * strd[strd.size() - 2] + weights_layout.spatial(1);
    }
    int32_t z = 1;
    if (spatial_dims > 2) {
        z = off_factor * pad[pad.size() - 3] + (input_layout.spatial(2) - 1) * strd[strd.size() - 3] + weights_layout.spatial(2);
    }

    tensor output_size(input_layout.batch(),
                       number_of_features, x, y, z);
    return {data_type, input_layout.format, output_size};
}

std::string deconvolution_inst::to_string(deconvolution_node const& node) {
    auto desc = node.get_primitive();
    auto strd = desc->stride;
    auto split = desc->split();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;
    std::stringstream ss_weights, ss_biases;

    for (size_t i = 0; i < desc->weights.size(); ++i) {
        ss_weights << node.weights(i).id();
        ss_weights << ", count: " << node.weights(i).get_output_layout().count();
        i != (desc->weights.size() - 1) ? ss_weights << ", " : ss_weights << "";
        if (node.get_depthwise_sep_opt())
            break;
    }

    for (size_t i = 0; i < desc->bias.size(); ++i) {
        ss_biases << node.bias(i).id();
        ss_biases << ", count: " << node.bias(i).get_output_layout().count();
        i != (desc->bias.size() - 1) ? ss_biases << ", " : ss_biases << "";
        if (node.get_depthwise_sep_opt())
            break;
    }

    json_composite deconv_info;
    deconv_info.add("weights count", desc->weights.size());
    deconv_info.add("bias count", desc->bias.size());
    deconv_info.add("stride", cldnn::to_string(strd));
    deconv_info.add("pad", cldnn::to_string(desc->pad));
    deconv_info.add("split", split);
    deconv_info.add("groups", desc->groups);
    if (desc->with_output_size) {
        json_composite ud_out_size_info;
        ud_out_size_info.add("size", desc->output_size.to_string());
        deconv_info.add("with_user_defined_output_size", ud_out_size_info);
    }
    node_info->add("deconvolution info", deconv_info);
    node_info->dump(primitive_description);
    return primitive_description.str();
}

deconvolution_inst::typed_primitive_inst(network& network, deconvolution_node const& node)
    : parent(network, node) {
    auto stride = argument.stride;
    auto pad = argument.pad;

    auto input_layout = node.input().get_output_layout();
    auto output_layout = node.get_output_layout();
    auto output_size = output_layout.size;

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input size",
                          input_layout.get_rank(),
                          "output size",
                          output_layout.get_rank(),
                          "Input/output number of dimension does not match.");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Stride size",
                          stride.size(),
                          "output size",
                          output_layout.get_spatial_rank(),
                          "Stride/output number of dimension does not match.");

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Input offset size",
                          pad.size(),
                          "input number of dimensions",
                          output_layout.get_spatial_rank(),
                          "");

    auto split = node.get_split();
    for (decltype(split) j = 0; j < split; j++) {
        auto filter_inst = node.weights(j).get_output_layout().convert_to_weights_layout(argument.grouped_weights_shape);

        if (argument.bias.size() != 0) {
            auto bias_inst = node.bias(j).get_output_layout();
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias batch[0]",
                                  bias_inst.batch(),
                                  "dimension size",
                                  1,
                                  "Batch[0] of bias should be 1. Bias isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias feature[0]",
                                  bias_inst.feature(),
                                  "output feature size / split",
                                  output_layout.feature(),
                                  "Biases/output feature maps number does not match.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[2]",
                                  bias_inst.spatial(2),
                                  "dimension size",
                                  1,
                                  "Spatial[2] of bias should be 1. Bias isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[1]",
                                  bias_inst.spatial(1),
                                  "dimension size",
                                  1,
                                  "Spatial[1] of bias should be 1. Bias isn't 1D vector.");
            CLDNN_ERROR_NOT_EQUAL(node.id(),
                                  "Bias spatial[0]",
                                  bias_inst.spatial(0),
                                  "dimension size",
                                  1,
                                  "Spatial[0] of bias should be 1. Bias isn't 1D vector.");
        }

        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "deconvolution padding filling value",
                              node.get_output_layout().data_padding.filling_value(),
                              "padding mode",
                              0.0f,
                              "Unknown padding mode in deconvolution.");
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Weights feature maps number",
                              filter_inst.ifm() * filter_inst.group(),
                              "input feature maps number",
                              input_layout.feature(),
                              "Weights/ifm mismatch");
    }
}
}  // namespace cldnn
