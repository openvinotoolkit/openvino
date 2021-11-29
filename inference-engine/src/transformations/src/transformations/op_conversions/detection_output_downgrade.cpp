// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/detection_output_downgrade.hpp"

#include <ngraph/op/util/detection_output_base.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::op::util;

NGRAPH_RTTI_DEFINITION(pass::ConvertDetectionOutput8ToDetectionOutput1,
                       "ConvertDetectionOutput8ToDetectionOutput1", 0);

pass::ConvertDetectionOutput8ToDetectionOutput1::
    ConvertDetectionOutput8ToDetectionOutput1() {
  MATCHER_SCOPE(ConvertDetectionOutput8ToDetectionOutput1);

  auto detection_output_v8_pattern =
      pattern::wrap_type<opset8::DetectionOutput>();

  matcher_pass_callback callback = [=](pattern::Matcher& m) {
    auto detection_output_v8_node =
        std::dynamic_pointer_cast<opset8::DetectionOutput>(m.get_match_root());
    if (!detection_output_v8_node) return false;
    const auto& attributes_v8 = detection_output_v8_node->get_attrs();
    auto num_classes =
        detection_output_v8_node->compute_num_classes(attributes_v8);

    // the transformation is applicable only if the number of classes is deduced
    if (num_classes.is_dynamic()) return false;

    opset1::DetectionOutput::Attributes attributes_v1 =
        static_cast<DetectionOutputBase::AttributesBase>(attributes_v8);
    attributes_v1.num_classes = num_classes.get_length();

    std::shared_ptr<opset1::DetectionOutput> detection_output_v1_node = nullptr;
    if (detection_output_v8_node->get_input_size() == 3) {
      detection_output_v1_node = make_shared<opset1::DetectionOutput>(
          detection_output_v8_node->input_value(0),
          detection_output_v8_node->input_value(1),
          detection_output_v8_node->input_value(2), attributes_v1);
    } else if (detection_output_v8_node->get_input_size() == 5) {
      detection_output_v1_node = make_shared<opset1::DetectionOutput>(
          detection_output_v8_node->input_value(0),
          detection_output_v8_node->input_value(1),
          detection_output_v8_node->input_value(2),
          detection_output_v8_node->input_value(3),
          detection_output_v8_node->input_value(4), attributes_v1);
    }
    if (!detection_output_v1_node) return false;

    detection_output_v1_node->set_friendly_name(
        detection_output_v8_node->get_friendly_name());
    ngraph::copy_runtime_info(detection_output_v8_node,
                              detection_output_v1_node);
    ngraph::replace_node(detection_output_v8_node, detection_output_v1_node);
    return true;
  };

  auto m =
      make_shared<pattern::Matcher>(detection_output_v8_pattern, matcher_name);
  register_matcher(m, callback);
}
