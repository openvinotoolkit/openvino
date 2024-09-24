// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/detection_output_upgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace std;
using namespace ov;
using namespace ov::op::util;

pass::ConvertDetectionOutput1ToDetectionOutput8::ConvertDetectionOutput1ToDetectionOutput8() {
    MATCHER_SCOPE(ConvertDetectionOutput1ToDetectionOutput8);

    auto detection_output_v1_pattern = pattern::wrap_type<ov::op::v0::DetectionOutput>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto detection_output_v1_node = ov::as_type_ptr<ov::op::v0::DetectionOutput>(m.get_match_root());
        if (!detection_output_v1_node)
            return false;

        const auto& attributes_v1 = detection_output_v1_node->get_attrs();
        ov::op::v8::DetectionOutput::Attributes attributes_v8;
        attributes_v8.background_label_id = attributes_v1.background_label_id;
        attributes_v8.clip_after_nms = attributes_v1.clip_after_nms;
        attributes_v8.clip_before_nms = attributes_v1.clip_before_nms;
        attributes_v8.code_type = attributes_v1.code_type;
        attributes_v8.confidence_threshold = attributes_v1.confidence_threshold;
        attributes_v8.decrease_label_id = attributes_v1.decrease_label_id;
        attributes_v8.input_height = attributes_v1.input_height;
        attributes_v8.input_width = attributes_v1.input_width;
        attributes_v8.keep_top_k = attributes_v1.keep_top_k;
        attributes_v8.nms_threshold = attributes_v1.nms_threshold;
        attributes_v8.normalized = attributes_v1.normalized;
        attributes_v8.objectness_score = attributes_v1.objectness_score;
        attributes_v8.share_location = attributes_v1.share_location;
        attributes_v8.top_k = attributes_v1.top_k;
        attributes_v8.variance_encoded_in_target = attributes_v1.variance_encoded_in_target;

        std::shared_ptr<ov::op::v8::DetectionOutput> detection_output_v8_node = nullptr;
        if (detection_output_v1_node->get_input_size() == 3) {
            detection_output_v8_node =
                make_shared<ov::op::v8::DetectionOutput>(detection_output_v1_node->input_value(0),
                                                         detection_output_v1_node->input_value(1),
                                                         detection_output_v1_node->input_value(2),
                                                         attributes_v8);
        } else if (detection_output_v1_node->get_input_size() == 5) {
            detection_output_v8_node =
                make_shared<ov::op::v8::DetectionOutput>(detection_output_v1_node->input_value(0),
                                                         detection_output_v1_node->input_value(1),
                                                         detection_output_v1_node->input_value(2),
                                                         detection_output_v1_node->input_value(3),
                                                         detection_output_v1_node->input_value(4),
                                                         attributes_v8);
        }
        if (!detection_output_v8_node)
            return false;

        detection_output_v8_node->set_friendly_name(detection_output_v1_node->get_friendly_name());
        ov::copy_runtime_info(detection_output_v1_node, detection_output_v8_node);
        ov::replace_node(detection_output_v1_node, detection_output_v8_node);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(detection_output_v1_pattern, matcher_name);
    register_matcher(m, callback);
}
