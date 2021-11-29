// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/detection_output_upgrade.hpp"

#include <ngraph/op/util/detection_output_base.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::op::util;

NGRAPH_RTTI_DEFINITION(pass::ConvertDetectionOutput1ToDetectionOutput8,
                       "ConvertDetectionOutput1ToDetectionOutput8", 0);

pass::ConvertDetectionOutput1ToDetectionOutput8::
    ConvertDetectionOutput1ToDetectionOutput8() {
  MATCHER_SCOPE(ConvertDetectionOutput1ToDetectionOutput8);

  auto detection_output_v1_pattern =
      pattern::wrap_type<opset1::DetectionOutput>();

  matcher_pass_callback callback = [=](pattern::Matcher& m) {
    auto detection_output_v1_node =
        std::dynamic_pointer_cast<opset1::DetectionOutput>(m.get_match_root());
    if (!detection_output_v1_node) return false;

    const auto& attributes_v1 = detection_output_v1_node->get_attrs();
    opset8::DetectionOutput::Attributes attributes_v8 =
        static_cast<DetectionOutputBase::AttributesBase>(attributes_v1);

    std::shared_ptr<opset8::DetectionOutput> detection_output_v8_node = nullptr;
    if (detection_output_v1_node->get_input_size() == 3) {
      auto detection_output_v8_node = make_shared<opset8::DetectionOutput>(
          detection_output_v1_node->input_value(0),
          detection_output_v1_node->input_value(1),
          detection_output_v1_node->input_value(2), attributes_v8);
    } else if (detection_output_v1_node->get_input_size() == 5) {
      auto detection_output_v8_node = make_shared<opset8::DetectionOutput>(
          detection_output_v1_node->input_value(0),
          detection_output_v1_node->input_value(1),
          detection_output_v1_node->input_value(2),
          detection_output_v1_node->input_value(3),
          detection_output_v1_node->input_value(4), attributes_v8);
    }
    if (!detection_output_v8_node) return false;

    detection_output_v8_node->set_friendly_name(
        detection_output_v1_node->get_friendly_name());
    ngraph::copy_runtime_info(detection_output_v1_node,
                              detection_output_v8_node);
    ngraph::replace_node(detection_output_v1_node, detection_output_v8_node);
    return true;
  };

  auto m = make_shared<pattern::Matcher>(detection_output_v1_pattern,
      matcher_name);
  register_matcher(m, callback);
}
