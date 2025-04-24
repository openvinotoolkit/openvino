// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output_inst.h"
#include "primitive_type_base.h"
#include "detection_output_shape_inference.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(detection_output)

layout detection_output_inst::calc_output_layout(detection_output_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for "
           "detection_output_node!");
    auto desc = impl_param.typed_desc<detection_output>();
    CLDNN_ERROR_NOT_EQUAL(desc->id,
                          "Detection output layer input number",
                          impl_param.input_layouts.size(),
                          "expected number of inputs",
                          static_cast<size_t>(3),
                          "");

    auto input_layout = impl_param.get_input_layout();

    // Batch size and feature size are 1.
    // Number of bounding boxes to be kept is set to keep_top_k*batch size.
    // If number of detections is lower than top_k, will write dummy results at the end with image_id=-1.
    // Each row is a 7 dimension vector, which stores:
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    int output_size = static_cast<int>(input_layout.get_linear_size()) / PRIOR_BOX_SIZE;
    int num_classes = desc->num_classes;

    if (desc->share_location) {
        num_classes = (desc->background_label_id == 0) ? desc->num_classes - 1
                                                       : desc->num_classes;
        output_size *= num_classes;
    }

    if (desc->top_k != -1) {
        int top_k = desc->top_k * num_classes * input_layout.batch();
        if (top_k < output_size) {
            output_size = top_k;
        }
    }

    output_size *= DETECTION_OUTPUT_ROW_SIZE;
    // Add space for number of output results per image - needed in the next detection output step
    output_size += ((input_layout.batch() + 15) / 16) * 16;

    return {input_layout.data_type, cldnn::format::bfyx,
            cldnn::tensor(1, 1, DETECTION_OUTPUT_ROW_SIZE, desc->keep_top_k * input_layout.batch())};
}

template<typename ShapeType>
std::vector<layout> detection_output_inst::calc_output_layouts(detection_output_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<detection_output>();
    auto input0_layout = impl_param.get_input_layout(0);
    auto box_logits_shape = input0_layout.get<ShapeType>();
    auto output_type = desc->output_data_types[0].value_or(input0_layout.data_type);
    auto output_format = input0_layout.format;

    ShapeType class_preds_shape = impl_param.get_input_layout(1).get<ShapeType>();
    ShapeType proposals_shape = impl_param.get_input_layout(2).get<ShapeType>();
    std::vector<ShapeType> output_shapes = { ShapeType() };
    std::vector<ShapeType> input_shapes = {
        box_logits_shape,
        class_preds_shape,
        proposals_shape
    };

    for (size_t i = 3; i < impl_param.input_layouts.size(); ++i) {
        input_shapes.push_back(impl_param.input_layouts[i].get<ShapeType>());
    }

    if (desc->num_classes == -1) {
        ov::op::v8::DetectionOutput op;
        ov::op::util::DetectionOutputBase::AttributesBase attrs;
        attrs.top_k = desc->top_k;
        attrs.variance_encoded_in_target = desc->variance_encoded_in_target;
        attrs.keep_top_k = { desc->keep_top_k };
        attrs.share_location = desc->share_location;
        attrs.normalized = desc->prior_is_normalized;
        op.set_attrs(attrs);

        output_shapes = ov::op::v8::shape_infer(&op, input_shapes);
    } else {
        ov::op::v0::DetectionOutput op;
        ov::op::v0::DetectionOutput::Attributes attrs;
        attrs.num_classes = desc->num_classes;
        attrs.top_k = desc->top_k;
        attrs.variance_encoded_in_target = desc->variance_encoded_in_target;
        attrs.keep_top_k = { desc->keep_top_k };
        attrs.share_location = desc->share_location;
        attrs.normalized = desc->prior_is_normalized;
        op.set_attrs(attrs);

        output_shapes = ov::op::v0::shape_infer(&op, input_shapes);
    }

    return { layout{output_shapes[0], output_type, output_format} };
}

template std::vector<layout> detection_output_inst::calc_output_layouts<ov::PartialShape>(detection_output_node const& node,
                                                                                          const kernel_impl_params& impl_param);

std::string detection_output_inst::to_string(detection_output_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto share_location = desc->share_location ? "true" : "false";
    auto variance_encoded = desc->variance_encoded_in_target ? "true" : "false";
    auto prior_is_normalized = desc->prior_is_normalized ? "true" : "false";
    auto decrease_label_id = desc->decrease_label_id ? "true" : "false";
    auto clip_before_nms = desc->clip_before_nms ? "true" : "false";
    auto clip_after_nms = desc->clip_after_nms ? "true" : "false";
    auto& input_location = node.location();
    auto& input_prior_box = node.prior_box();
    auto& input_confidence = node.confidence();

    std::stringstream primitive_description;
    std::string str_code_type;

    switch (desc->code_type) {
        case prior_box_code_type::corner:
            str_code_type = "corner";
            break;
        case prior_box_code_type::center_size:
            str_code_type = "center size";
            break;
        case prior_box_code_type::corner_size:
            str_code_type = "corner size";
            break;
        default:
            str_code_type = "not supported code type";
            break;
    }

    json_composite detec_out_info;
    detec_out_info.add("input location id", input_location.id());
    detec_out_info.add("input confidence id", input_confidence.id());
    detec_out_info.add("input prior box id", input_prior_box.id());
    detec_out_info.add("num_classes:", desc->num_classes);
    detec_out_info.add("keep_top_k", desc->keep_top_k);
    detec_out_info.add("share_location", share_location);
    detec_out_info.add("background_label_id", desc->background_label_id);
    detec_out_info.add("nms_treshold", desc->nms_threshold);
    detec_out_info.add("top_k", desc->top_k);
    detec_out_info.add("eta", desc->eta);
    detec_out_info.add("code_type", str_code_type);
    detec_out_info.add("variance_encoded", variance_encoded);
    detec_out_info.add("confidence_threshold", desc->confidence_threshold);
    detec_out_info.add("prior_info_size", desc->prior_info_size);
    detec_out_info.add("prior_coordinates_offset", desc->prior_coordinates_offset);
    detec_out_info.add("prior_is_normalized", prior_is_normalized);
    detec_out_info.add("input_width", desc->input_width);
    detec_out_info.add("input_height", desc->input_height);
    detec_out_info.add("decrease_label_id", decrease_label_id);
    detec_out_info.add("clip_before_nms", clip_before_nms);
    detec_out_info.add("clip_after_nms", clip_after_nms);
    detec_out_info.add("objectness_score", desc->objectness_score);
    detec_out_info.dump(primitive_description);

    node_info->add("dection output info", detec_out_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

detection_output_inst::typed_primitive_inst(network& network, detection_output_node const& node)
    : parent(network, node) {
    auto location_layout = node.location().get_output_layout();
    auto confidence_layout = node.confidence().get_output_layout();
    auto prior_box_layout = node.prior_box().get_output_layout();
    if (location_layout.is_dynamic() || confidence_layout.is_dynamic() || prior_box_layout.is_dynamic())
        return;

    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "Location memory format",
                                  location_layout.format.value,
                                  "expected bfyx input format",
                                  format::bfyx);
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "Confidence memory format",
                                  confidence_layout.format.value,
                                  "expected bfyx input format",
                                  format::bfyx);
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(),
                                  "Prior box memory format",
                                  prior_box_layout.format.value,
                                  "expected bfyx input format",
                                  format::bfyx);

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Location input dimensions",
                          (location_layout.feature() * location_layout.batch()),
                          "detection output layer dimensions",
                          static_cast<int>(location_layout.count()),
                          "Location input/ detection output dims mismatch");

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Confidence input dimensions",
                          (confidence_layout.feature() * confidence_layout.batch()),
                          "detection output layer dimensions",
                          static_cast<int>(confidence_layout.count()),
                          "Confidence input/detection output dims mistmach");

    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Confidence batch size",
                          confidence_layout.batch(),
                          "location input batch size",
                          location_layout.batch(),
                          "Batch sizes mismatch.");

    auto desc = node.get_primitive();
    int prior_feature_size = desc->variance_encoded_in_target ? 1 : 2;
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Prior box spatial X", prior_box_layout.spatial(0), "expected value", 1, "");
    CLDNN_ERROR_NOT_EQUAL(node.id(),
                          "Prior box feature size",
                          prior_box_layout.feature(),
                          "expected value",
                          prior_feature_size,
                          "");

    CLDNN_ERROR_BOOL(node.id(),
                     "Detection output layer padding",
                     node.is_padded(),
                     "Detection output layer doesn't support output padding.");
    CLDNN_ERROR_BOOL(node.id(),
                     "Detection output layer Prior-box input padding",
                     node.get_dependency(2).is_padded(),
                     "Detection output layer doesn't support input padding in Prior-Box input");
}
}  // namespace cldnn
