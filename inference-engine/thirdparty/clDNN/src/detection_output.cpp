/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "detection_output_inst.h"
#include "primitive_type_base.h"
#include "network_impl.h"
#include "error_handler.h"
#include "json_object.h"

namespace cldnn
{
primitive_type_id detection_output_type_id()
{
    static primitive_type_base<detection_output> instance;
    return &instance;
}

layout detection_output_inst::calc_output_layout(detection_output_node const& node)
{
    assert((bool)node.get_primitive()->output_data_type == false
           && "Output data type forcing is not supported for "
              "detection_output_node!");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Detection output layer input number", node.get_dependencies().size(), "expected number of inputs", static_cast<size_t>(3), "");

    auto input_layout = node.location().get_output_layout();

    // Batch size and feature size are 1.
    // Number of bounding boxes to be kept is set to keep_top_k*batch size.
    // If number of detections is lower than top_k, will write dummy results at the end with image_id=-1.
    // Each row is a 7 dimension vector, which stores:
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    int output_size = (int)input_layout.get_linear_size() / PRIOR_BOX_SIZE;
    int num_classes = node.get_primitive()->num_classes;

    if (node.get_primitive()->share_location)
    {
        num_classes = (node.get_primitive()->background_label_id == 0) ? node.get_primitive()->num_classes - 1 : node.get_primitive()->num_classes;
        output_size *= num_classes;
    }

    if (node.get_primitive()->top_k != -1)
    {
        int top_k = node.get_primitive()->top_k * num_classes * input_layout.size.batch[0];
        if (top_k < output_size)
        {
            output_size = top_k;
        }
    }

    output_size *= DETECTION_OUTPUT_ROW_SIZE;
    // Add space for number of output results per image - needed in the next detection output step
    output_size += ((input_layout.size.batch[0] + 15) / 16) * 16;

    if (node.get_program().get_options().get<build_option_type::detection_output_gpu>()->enabled())
    {
        return{ input_layout.data_type, cldnn::format::bfyx, cldnn::tensor(1, 1, 1, output_size) };
    }
    else
    {
        return{ input_layout.data_type, cldnn::format::bfyx, cldnn::tensor(1, 1, DETECTION_OUTPUT_ROW_SIZE, node.get_primitive()->keep_top_k * input_layout.size.batch[0]) };
    }
}

std::string detection_output_inst::to_string(detection_output_node const& node)
{
    auto node_info           = node.desc_to_json();
    auto desc                = node.get_primitive();
    auto share_location      = desc->share_location ? "true" : "false";
    auto variance_encoded    = desc->variance_encoded_in_target ? "true" : "false";
    auto prior_is_normalized = desc->prior_is_normalized ? "true" : "false";
    auto decrease_label_id   = desc->decrease_label_id ? "true" : "false";
    auto clip_before_nms     = desc->clip_before_nms ? "true" : "false";
    auto clip_after_nms      = desc->clip_after_nms ? "true" : "false";
    auto& input_location     = node.location();
    auto& input_prior_box    = node.prior_box();
    auto& input_confidence   = node.confidence();


    std::stringstream primitive_description;
    std::string       str_code_type;

    switch (desc->code_type)
    {
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
    detec_out_info.dump(primitive_description);

    node_info->add("dection output info", detec_out_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

detection_output_inst::typed_primitive_inst(network_impl& network, detection_output_node const& node)
    :parent(network, node)
{
    auto location_layout = node.location().get_output_layout();
    auto confidence_layout = node.confidence().get_output_layout();
    auto prior_box_layout = node.prior_box().get_output_layout();
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Location memory format", location_layout.format.value, "expected bfyx input format", format::bfyx );
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Confidence memory format", confidence_layout.format.value, "expected bfyx input format", format::bfyx );
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Prior box memory format", prior_box_layout.format.value, "expected bfyx input format", format::bfyx );

    tensor location_size = location_layout.size;
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Location input dimensions", (location_size.feature[0] * location_size.batch[0]), "detection output layer dimensions", (int)location_layout.count(), "Location input/ detection output dims mismatch");

    tensor confidence_size = confidence_layout.size;
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Confidence input dimensions", (confidence_size.feature[0] * confidence_size.batch[0]), "detection output layer dimensions", (int)confidence_layout.count(), "Confidence input/detection output dims mistmach");

    CLDNN_ERROR_NOT_EQUAL(node.id(), "Confidence batch size", confidence_size.batch[0], "location input batch size", location_size.batch[0], "Batch sizes mismatch.");

    auto desc              = node.get_primitive();
    int prior_feature_size = desc->variance_encoded_in_target ? 1 : 2;
    tensor prior_box_size = prior_box_layout.size;
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Prior box spatial X", prior_box_size.spatial[0], "expected value", 1, "");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Prior box feature size", prior_box_size.feature[0], "expected value", prior_feature_size, "");

    CLDNN_ERROR_BOOL(node.id(), "Detection output layer padding", node.is_padded(), "Detection output layer doesn't support output padding.");
    CLDNN_ERROR_BOOL(node.id(), "Detection output layer Prior-box input padding", node.get_dependency(2).is_padded(), "Detection output layer doesn't support input padding in Prior-Box input");
}

/************************ Detection Output keep_top_k part ************************/

primitive_type_id detection_output_sort_type_id()
{
    static primitive_type_base<detection_output_sort> instance;
    return &instance;
}

layout detection_output_sort_inst::calc_output_layout(detection_output_sort_node const& node)
{
    assert((bool)node.get_primitive()->output_data_type == false
           && "Output data type forcing is not supported for "
              "detection_output_sort_node!");
    CLDNN_ERROR_NOT_EQUAL(node.id(), "Detection output layer input number", node.get_dependencies().size(), "expected number of inputs", static_cast<size_t>(1), "");

    auto input_layout = node.input().get_output_layout();
    int keep_top_k = node.as<detection_output_sort>().get_primitive()->keep_top_k;
    int num_images = node.as<detection_output_sort>().get_primitive()->num_images;

    // If detection output sort is used as a second part of detection output get proper info from detection otput node
    if (num_images == 0)
    {
        CLDNN_ERROR_BOOL(node.id(), "node.get_dependency(0).is_type<detection_output>()", !node.get_dependency(0).is_type<detection_output>(), "Cannot calculate output layout.");
        input_layout = node.get_dependency(0).as<detection_output>().location().get_output_layout();
        keep_top_k = node.get_dependency(0).as<detection_output>().get_primitive()->keep_top_k;
        num_images = input_layout.size.batch[0];
    }
    // Batch size and feature size are 1.
    // Number of bounding boxes to be kept is set to keep_top_k*batch size.
    // If number of detections is lower than keep_top_k, will write dummy results at the end with image_id=-1.
    // Each row is a 7 dimension vector, which stores:
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    return{ input_layout.data_type, cldnn::format::bfyx, cldnn::tensor(1, 1, DETECTION_OUTPUT_ROW_SIZE, keep_top_k * num_images) };
}

std::string detection_output_sort_inst::to_string(detection_output_sort_node const& node)
{
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();

    auto& input_bboxes = node.input();

    std::stringstream primitive_description;

    json_composite detec_out_info;
    detec_out_info.add("input bboxes id", input_bboxes.id());
    detec_out_info.add("num_classes:", desc->num_images);
    detec_out_info.add("num_classes:", desc->num_classes);
    detec_out_info.add("keep_top_k", desc->keep_top_k);
    detec_out_info.add("share_location", desc->share_location);
    detec_out_info.add("top_k", desc->top_k);
    detec_out_info.dump(primitive_description);

    node_info->add("dection output info", detec_out_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

detection_output_sort_inst::typed_primitive_inst(network_impl& network, detection_output_sort_node const& node)
    :parent(network, node)
{
    CLDNN_ERROR_NOT_PROPER_FORMAT(node.id(), "Input memory format", node.get_dependency(0).get_output_layout().format.value, "expected bfyx input format", format::bfyx);

    CLDNN_ERROR_BOOL(node.id(), "Detecion output layer padding", node.is_padded(), "Detection output layer doesn't support output padding.");
}
}
