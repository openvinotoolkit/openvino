// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/retinanet_nms_replacement.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

namespace {

// Helper function to find node by name pattern
shared_ptr<Node> find_node_by_name(const shared_ptr<ov::Model>& model, const string& name_pattern) {
    for (const auto& node : model->get_ordered_ops()) {
        const auto& name = node->get_friendly_name();
        if (name.find(name_pattern) != string::npos) {
            return node;
        }
    }
    return nullptr;
}

// Helper function to find node with exact name
shared_ptr<Node> find_node_exact_name(const shared_ptr<ov::Model>& model, const string& name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == name) {
            return node;
        }
    }
    return nullptr;
}

// Helper function to find Parameter node (input placeholder)
shared_ptr<v0::Parameter> find_input_parameter(const shared_ptr<ov::Model>& model) {
    const auto& params = model->get_parameters();
    // Find the image input parameter (should have 4D shape: [batch, height, width, channels])
    for (const auto& param : params) {
        const auto& shape = param->get_partial_shape();
        if (shape.rank().is_static() && shape.rank().get_length() == 4) {
            return param;
        }
    }
    return params.empty() ? nullptr : params[0];
}

// Helper function to find NMS node and extract iou_threshold
float find_nms_iou_threshold(const shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (auto nms = dynamic_pointer_cast<v5::NonMaxSuppression>(node)) {
            // Try to get iou_threshold from input 3
            if (nms->get_input_size() > 3) {
                if (auto const_node = dynamic_pointer_cast<v0::Constant>(
                        nms->input_value(3).get_node_shared_ptr())) {
                    return const_node->cast_vector<float>()[0];
                }
            }
        }
        // Also check for other NMS versions
        if (auto nms = dynamic_pointer_cast<v9::NonMaxSuppression>(node)) {
            if (nms->get_input_size() > 3) {
                if (auto const_node = dynamic_pointer_cast<v0::Constant>(
                        nms->input_value(3).get_node_shared_ptr())) {
                    return const_node->cast_vector<float>()[0];
                }
            }
        }
    }
    // Default iou_threshold if not found
    return 0.5f;
}

// Build scales for prior boxes: [1/w, 1/h, 1/w, 1/h]
Output<Node> build_placeholder_scales(const shared_ptr<v0::Parameter>& placeholder) {
    const auto& name = placeholder->get_friendly_name();

    // Get shape of input image [batch, height, width, channels]
    auto shape = make_shared<v3::ShapeOf>(placeholder, element::i32);

    // Extract spatial dimensions [height, width] from shape (indices 1 and 2)
    // In OV StridedSlice: mask=0 means use the value, mask=1 means ignore
    auto begin = v0::Constant::create(element::i32, Shape{1}, {1});
    auto end = v0::Constant::create(element::i32, Shape{1}, {3});
    auto stride = v0::Constant::create(element::i32, Shape{1}, {1});

    auto spatial = make_shared<v1::StridedSlice>(
        shape, begin, end, stride,
        vector<int64_t>{0},  // begin_mask: use begin value 1
        vector<int64_t>{0}   // end_mask: use end value 3
    );

    // Convert to float and compute 1/spatial
    auto spatial_float = make_shared<v0::Convert>(spatial, element::f32);
    auto power = v0::Constant::create(element::f32, Shape{1}, {-1.0f});
    auto spatial_scale = make_shared<v1::Power>(spatial_float, power);

    // Reverse order to get [1/w, 1/h]
    auto order = v0::Constant::create(element::i32, Shape{2}, {1, 0});
    auto axis_const = v0::Constant::create(element::i32, Shape{}, {0});
    auto reverse = make_shared<v8::Gather>(spatial_scale, order, axis_const);

    // Concat to get [1/w, 1/h, 1/w, 1/h]
    auto priors_scale = make_shared<v0::Concat>(OutputVector{reverse, reverse}, 0);
    priors_scale->set_friendly_name(name + "/priors_scale");

    return priors_scale->output(0);
}

// Append variances to priors: creates tensor with shape [1, 2, num_priors*4]
// First row: priors, Second row: variances (tiled)
Output<Node> append_variances(const Output<Node>& priors_scale_node, const vector<float>& variance) {
    const auto& name = priors_scale_node.get_node()->get_friendly_name();

    // Get shape of priors [1, num_priors, 4]
    auto sp_shape = make_shared<v3::ShapeOf>(priors_scale_node, element::i32);

    // Get the -2 dimension (number of priors) from shape
    // Shape is [3], we want element at index 1 (or -2 from end)
    // StridedSlice [-2:-1] gives us [num_priors]
    // In OV: mask=0 means use the value, mask=1 means ignore (use boundary)
    auto begin = v0::Constant::create(element::i32, Shape{1}, {-2});
    auto end = v0::Constant::create(element::i32, Shape{1}, {-1});
    auto stride = v0::Constant::create(element::i32, Shape{1}, {1});

    auto shape_part = make_shared<v1::StridedSlice>(
        sp_shape, begin, end, stride,
        vector<int64_t>{0},  // begin_mask: use begin value -2
        vector<int64_t>{0}   // end_mask: use end value -1
    );

    // Create shape for tiling: [num_priors, 4]
    auto four_const = v0::Constant::create(element::i32, Shape{1}, {4});
    auto shape_for_tiling = make_shared<v0::Concat>(OutputVector{shape_part, four_const}, 0);

    // Create variance constant and broadcast to [num_priors, 4]
    auto variance_const = v0::Constant::create(element::f32, Shape{4}, variance);
    auto tile = make_shared<v3::Broadcast>(variance_const, shape_for_tiling);

    // Reshape priors to [-1, 4] (removes batch dimension, shape becomes [num_priors, 4])
    auto reshape_dim = v0::Constant::create(element::i32, Shape{2}, {-1, 4});
    auto sp_reshape = make_shared<v1::Reshape>(priors_scale_node, reshape_dim, false);

    // Concat priors [num_priors, 4] and variances [num_priors, 4] along axis 0
    // Result shape: [2*num_priors, 4]
    auto concat = make_shared<v0::Concat>(OutputVector{sp_reshape, tile}, 0);

    // Reshape to [1, 2, -1] for DetectionOutput format
    auto output_dims = v0::Constant::create(element::i32, Shape{3}, {1, 2, -1});
    auto output_node = make_shared<v1::Reshape>(concat, output_dims, false);
    output_node->set_friendly_name(name + "/priors_with_variances");

    return output_node->output(0);
}

}  // namespace

bool RetinaNetNmsToDetectionOutput::run_on_model(const shared_ptr<ov::Model>& model) {
    // RetinaNet-specific node names from retinanet.json
    const string regression_name = "regression/concat";
    const string classification_name = "classification/concat";
    const string anchors_name = "anchors/concat";

    // Output node name patterns
    const string output_pattern_1 = "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3";
    const string output_pattern_2 = "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3";
    const string output_pattern_3 = "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3";

    // Try to find input nodes
    auto regression_node = find_node_exact_name(model, regression_name);
    auto classification_node = find_node_exact_name(model, classification_name);
    auto anchors_node = find_node_exact_name(model, anchors_name);

    // If not found with exact names, try pattern matching
    if (!regression_node) {
        regression_node = find_node_by_name(model, "regression/concat");
    }
    if (!classification_node) {
        classification_node = find_node_by_name(model, "classification/concat");
    }
    if (!anchors_node) {
        anchors_node = find_node_by_name(model, "anchors/concat");
    }

    // Check if this is a RetinaNet model
    if (!regression_node || !classification_node || !anchors_node) {
        // Not a RetinaNet model, skip transformation
        return false;
    }

    // Find output nodes
    auto output_node_1 = find_node_by_name(model, output_pattern_1);
    auto output_node_2 = find_node_by_name(model, output_pattern_2);
    auto output_node_3 = find_node_by_name(model, output_pattern_3);

    if (!output_node_1 || !output_node_2 || !output_node_3) {
        // Output nodes not found, cannot apply transformation
        return false;
    }

    // Find input placeholder for computing scales
    auto placeholder = find_input_parameter(model);
    if (!placeholder) {
        return false;
    }

    // Extract iou_threshold from NMS node
    float iou_threshold = find_nms_iou_threshold(model);

    // RetinaNet custom attributes from retinanet.json
    const vector<float> variance = {0.2f, 0.2f, 0.2f, 0.2f};
    const float confidence_threshold = 0.05f;
    const int top_k = 6000;
    const int keep_top_k = 300;
    const int background_label_id = 1000;

    // Get input tensors
    auto regression_output = regression_node->output(0);
    auto classification_output = classification_node->output(0);
    auto anchors_output = anchors_node->output(0);

    // Take first batch slice from anchors (same anchors for all batches)
    // anchors shape: [batch, num_anchors, 4]
    // We want: [0:1, :, :] = first batch, all anchors, all coords
    // In OV StridedSlice: mask bit 1 means "ignore this value and use tensor boundary"
    // begin_mask=[0,0,0] - use begin values [0,0,0]
    // end_mask=[0,1,1] - use end[0]=1 for batch, ignore end for dims 1,2 (use tensor end)
    auto batch_begin = v0::Constant::create(element::i32, Shape{3}, {0, 0, 0});
    auto batch_end = v0::Constant::create(element::i32, Shape{3}, {1, 0, 0});
    auto batch_stride = v0::Constant::create(element::i32, Shape{3}, {1, 1, 1});

    auto priors_node = make_shared<v1::StridedSlice>(
        anchors_output, batch_begin, batch_end, batch_stride,
        vector<int64_t>{0, 0, 0},  // begin_mask: use begin values
        vector<int64_t>{0, 1, 1}   // end_mask: use end[0]=1, ignore end[1,2] (take to end)
    );
    priors_node->set_friendly_name("anchors/first_batch");

    // Build scales for prior boxes: [1/w, 1/h, 1/w, 1/h] shape [4]
    auto scales = build_placeholder_scales(placeholder);

    // Reshape scales to [1, 1, 4] for numpy-style broadcasting with priors [1, num_anchors, 4]
    // Multiply will automatically broadcast [1, 1, 4] to [1, num_anchors, 4]
    auto scales_reshape_dim = v0::Constant::create(element::i32, Shape{3}, {1, 1, 4});
    auto scales_reshaped = make_shared<v1::Reshape>(scales, scales_reshape_dim, false);
    scales_reshaped->set_friendly_name("scales_reshaped");

    // Scale priors to [0, 1] interval - Multiply auto-broadcasts
    auto priors_scaled = make_shared<v1::Multiply>(priors_node, scales_reshaped);
    priors_scaled->set_friendly_name("scaled_priors");

    // Append variances to priors
    auto priors_with_variances = append_variances(priors_scaled->output(0), variance);

    // Calculate prior boxes widths and heights
    auto split_axis = v0::Constant::create(element::i32, Shape{}, {2});
    auto split_lengths = v0::Constant::create(element::i32, Shape{4}, {1, 1, 1, 1});
    auto split_node = make_shared<v1::VariadicSplit>(priors_scaled, split_axis, split_lengths);

    // widths = x2 - x1, heights = y2 - y1
    auto priors_width = make_shared<v1::Subtract>(split_node->output(2), split_node->output(0));
    auto priors_height = make_shared<v1::Subtract>(split_node->output(3), split_node->output(1));

    // Concat widths and heights: [w, h, w, h]
    auto concat_wh_1 = make_shared<v0::Concat>(OutputVector{priors_width, priors_height}, -1);
    auto concat_wh_2 = make_shared<v0::Concat>(OutputVector{concat_wh_1, priors_width}, -1);
    auto concat_width_height = make_shared<v0::Concat>(OutputVector{concat_wh_2, priors_height}, -1);
    concat_width_height->set_friendly_name("priors_width_height");

    // Multiply regressions by widths/heights
    auto applied_regressions = make_shared<v1::Multiply>(concat_width_height, regression_output);
    applied_regressions->set_friendly_name("applied_regressions");

    // Reshape regression to 2D for DetectionOutput: [batch, -1]
    auto reshape_dim_2d = v0::Constant::create(element::i32, Shape{2}, {0, -1});
    auto reshape_regression = make_shared<v1::Reshape>(applied_regressions, reshape_dim_2d, true);
    reshape_regression->set_friendly_name("reshape_regression");

    // Reshape classification to 2D for DetectionOutput: [batch, -1]
    auto reshape_classes = make_shared<v1::Reshape>(classification_output, reshape_dim_2d, true);
    reshape_classes->set_friendly_name("reshape_classes");

    // Create DetectionOutput attributes
    v8::DetectionOutput::Attributes attrs;
    attrs.background_label_id = background_label_id;
    attrs.top_k = top_k;
    attrs.keep_top_k = {keep_top_k};
    attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs.variance_encoded_in_target = false;
    attrs.nms_threshold = iou_threshold;
    attrs.confidence_threshold = confidence_threshold;
    attrs.clip_after_nms = true;
    attrs.clip_before_nms = false;
    attrs.normalized = true;
    attrs.share_location = true;
    attrs.decrease_label_id = false;

    // Create DetectionOutput operation
    auto detection_output = make_shared<v8::DetectionOutput>(
        reshape_regression->output(0),
        reshape_classes->output(0),
        priors_with_variances,
        attrs
    );
    detection_output->set_friendly_name("detection_output");

    // Set output tensor name for the DetectionOutput
    detection_output->output(0).set_names({"detection_output:0"});

    // Find and replace Result nodes connected to output_node_1, output_node_2, output_node_3
    // We need to consolidate 3 outputs into 1 DetectionOutput
    bool replaced = false;
    vector<shared_ptr<v0::Result>> results_to_remove;

    for (const auto& result : model->get_results()) {
        auto parent = result->input_value(0).get_node_shared_ptr();
        if (parent == output_node_1 || parent == output_node_2 || parent == output_node_3) {
            results_to_remove.push_back(result);
        }
    }

    if (!results_to_remove.empty()) {
        // Replace the first result's input with detection_output
        results_to_remove[0]->input(0).replace_source_output(detection_output->output(0));
        replaced = true;

        // Remove extra results (DetectionOutput has single output instead of 3)
        for (size_t i = 1; i < results_to_remove.size(); ++i) {
            model->remove_result(results_to_remove[i]);
        }
    }

    // If we couldn't replace via Results, try replacing the nodes directly
    if (!replaced) {
        // Replace uses of output nodes with detection_output
        for (auto& target : output_node_1->output(0).get_target_inputs()) {
            target.replace_source_output(detection_output->output(0));
            replaced = true;
        }
    }

    return replaced;
}

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
