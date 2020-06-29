// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph/graph_util.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "transformations/convert_nms_4_to_nms_dynamic.hpp"

void ngraph::pass::UpgradeNMS4ToNMSDynamic::upgrade_nms4_to_nms_dynamic() {
    auto boxes = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1000, 4});
    auto scores = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1000});
    auto max_output_boxes_per_class = ngraph::opset4::Constant::create(element::i64, Shape{}, {10});
    auto iou_threshold = ngraph::opset4::Constant::create(element::f32, Shape{}, {0.75});
    auto score_threshold = ngraph::opset4::Constant::create(element::f32, Shape{}, {0.7});
    auto nms = std::make_shared<ngraph::opset4::NonMaxSuppression>(boxes, scores, max_output_boxes_per_class,
                                                                   iou_threshold, score_threshold);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher &m) {
        auto nms_4 = std::dynamic_pointer_cast<ngraph::opset4::NonMaxSuppression>(m.get_match_root());
        if (!nms_4) {
            return false;
        }

        const auto box_encoding = static_cast<const op::dynamic::NonMaxSuppression::BoxEncodingType>(nms_4->get_box_encoding());
        const auto new_args = nms_4->input_values();
        const auto& arg2 = new_args.size() > 2 ? new_args.at(2) : ngraph::opset4::Constant::create(element::i32, Shape{}, {0});
        const auto& arg3 = new_args.size() > 3 ? new_args.at(3) : ngraph::opset4::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = new_args.size() > 4 ? new_args.at(4) : ngraph::opset4::Constant::create(element::f32, Shape{}, {.0f});

        const auto nms_dynamic = std::make_shared<op::dynamic::NonMaxSuppression>(
                new_args.at(0),
                new_args.at(1),
                arg2,
                arg3,
                arg4,
                box_encoding,
                nms_4->get_sort_result_descending(),
                nms_4->get_output_type());

        nms_dynamic->set_friendly_name(nms_4->get_friendly_name());
        ngraph::copy_runtime_info(nms_4, nms_dynamic);
        ngraph::replace_node(nms_4, nms_dynamic);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "UpgradeNMS4ToDynamic");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

bool ngraph::pass::UpgradeNMS4ToNMSDynamic::run_on_function(std::shared_ptr<ngraph::Function> f) {
    return ngraph::op::util::has_op_with_type<ngraph::opset4::NonZero>(f);
}
