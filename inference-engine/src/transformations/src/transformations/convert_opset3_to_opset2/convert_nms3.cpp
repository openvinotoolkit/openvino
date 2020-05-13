// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset3_to_opset2/convert_nms3.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertNMS3::convert_nms3() {
    auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
    auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
    auto max_output_boxes_per_class = ngraph::opset3::Constant::create(element::i64, Shape{}, {10});
    auto iou_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.75});
    auto score_threshold = ngraph::opset3::Constant::create(element::f32, Shape{}, {0.7});
    auto nms = std::make_shared<ngraph::opset3::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                   iou_threshold, score_threshold);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto nms = std::dynamic_pointer_cast<ngraph::opset3::NonMaxSuppression>(m.get_match_root());
        if (!nms) {
            return false;
        }

        Output<Node> last;
        ngraph::NodeVector new_ops;

        auto new_nms = std::make_shared<ngraph::opset2::NonMaxSuppression>(nms->input_value(0), nms->input_value(1),
                nms->input_value(2), nms->input_value(3), nms->input_value(4),
                static_cast<const op::v1::NonMaxSuppression::BoxEncodingType>(nms->get_box_encoding()),
                nms->get_sort_result_descending());

        new_ops.push_back(new_nms);
        // if the output is the i32 then it matches behavior of the v1::NonMaxSuppression otherwise need to insert Convert
        if (nms->get_output_type() == element::i32) {
            last = new_nms;
        } else {
            last = std::make_shared<ngraph::opset2::Convert>(new_nms, nms->get_output_type());
            new_ops.push_back(last.get_node_shared_ptr());
        }

        last.get_node_shared_ptr()->set_friendly_name(nms->get_friendly_name());
        ngraph::copy_runtime_info(nms, new_ops);
        ngraph::replace_node(nms, last.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS3");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
