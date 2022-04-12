// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_nms9_to_nms_ie_internal.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "ngraph_ops/nms_ie_internal.hpp"
#include "transformations/utils/utils.hpp"

ngraph::pass::ConvertNMS9ToNMSIEInternal::ConvertNMS9ToNMSIEInternal() {
    MATCHER_SCOPE(ConvertNMS9ToNMSIEInternal);
    auto nms = ngraph::pattern::wrap_type<ngraph::opset9::NonMaxSuppression>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto nms_9 = std::dynamic_pointer_cast<ngraph::opset9::NonMaxSuppression>(m.get_match_root());
        if (!nms_9 || transformation_callback(nms_9)) {
            return false;
        }

        const auto new_args = nms_9->input_values();
        const std::size_t num_of_inputs = new_args.size();

        const auto& arg2 =
            num_of_inputs > 2 ? new_args.at(2) : ngraph::opset9::Constant::create(element::i32, Shape{}, {0});
        const auto& arg3 =
            num_of_inputs > 3 ? new_args.at(3) : ngraph::opset9::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 =
            num_of_inputs > 4 ? new_args.at(4) : ngraph::opset9::Constant::create(element::f32, Shape{}, {.0f});

        // vector of new nGraph operations
        NodeVector new_ops;

        auto one_dim_shape = Shape{1};

        Output<Node> new_max_per_class;
        Output<Node> new_iou_threshold;
        Output<Node> new_score_threshold;
        Output<Node> new_soft_nms_sigma;

        Output<Node> new_shape_for_max_per_class = opset1::Constant::create(ngraph::element::i64, Shape{1}, {1});
        Output<Node> new_shape_for_iou_threshold = opset1::Constant::create(ngraph::element::i64, Shape{1}, {1});
        Output<Node> new_shape_for_score_threshold = opset1::Constant::create(ngraph::element::i64, Shape{1}, {1});
        Output<Node> new_shape_for_soft_nms_sigma = opset1::Constant::create(ngraph::element::i64, Shape{1}, {1});

        new_max_per_class = std::make_shared<opset1::Reshape>(arg2, new_shape_for_max_per_class, true);
        new_ops.emplace_back(new_max_per_class.get_node_shared_ptr());

        new_iou_threshold = std::make_shared<opset1::Reshape>(arg3, new_shape_for_iou_threshold, true);
        new_ops.emplace_back(new_iou_threshold.get_node_shared_ptr());

        new_score_threshold = std::make_shared<opset1::Reshape>(arg4, new_shape_for_score_threshold, true);
        new_ops.emplace_back(new_score_threshold.get_node_shared_ptr());

        int center_point_box = 0;
        switch (nms_9->get_box_encoding()) {
        case ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CENTER:
            center_point_box = 1;
            break;
        case ::ngraph::opset9::NonMaxSuppression::BoxEncodingType::CORNER:
            center_point_box = 0;
            break;
        default:
            throw ngraph_error("NonMaxSuppression layer " + nms_9->get_friendly_name() +
                               " has unsupported box encoding");
        }

        std::shared_ptr<op::internal::NonMaxSuppressionIEInternal> nms_legacy{nullptr};

        if (num_of_inputs > 5 && !nms_9->is_soft_nms_sigma_constant_and_default()) {
            new_soft_nms_sigma = std::make_shared<opset1::Reshape>(new_args.at(5), new_shape_for_soft_nms_sigma, true);
            new_ops.emplace_back(new_soft_nms_sigma.get_node_shared_ptr());
            nms_legacy =
                std::make_shared<op::internal::NonMaxSuppressionIEInternal>(new_args.at(0),
                                                                            new_args.at(1),
                                                                            new_max_per_class,
                                                                            new_iou_threshold,
                                                                            new_score_threshold,
                                                                            new_soft_nms_sigma,
                                                                            center_point_box,
                                                                            nms_9->get_sort_result_descending(),
                                                                            element::i32,
                                                                            nms_9->get_output_element_type(1));
            new_ops.push_back(nms_legacy);
        } else {
            nms_legacy =
                std::make_shared<op::internal::NonMaxSuppressionIEInternal>(new_args.at(0),
                                                                            new_args.at(1),
                                                                            new_max_per_class,
                                                                            new_iou_threshold,
                                                                            new_score_threshold,
                                                                            center_point_box,
                                                                            nms_9->get_sort_result_descending(),
                                                                            element::i32,
                                                                            nms_9->get_output_element_type(1));
            new_ops.push_back(nms_legacy);
        }

        Output<Node> output_0 = nms_legacy->output(0);
        if (nms_9->output(0).get_element_type() != output_0.get_element_type()) {
            output_0 = std::make_shared<opset1::Convert>(output_0, nms_9->output(0).get_element_type());
            output_0.get_node_shared_ptr()->set_friendly_name(op::util::create_ie_output_name(nms_9->output(0)));
            new_ops.emplace_back(output_0.get_node_shared_ptr());
        }

        Output<Node> output_2 = nms_legacy->output(2);
        if (nms_9->output(2).get_element_type() != output_2.get_element_type()) {
            output_2 = std::make_shared<opset1::Convert>(output_2, nms_9->output(2).get_element_type());
            output_2.get_node_shared_ptr()->set_friendly_name(op::util::create_ie_output_name(nms_9->output(2)));
            new_ops.emplace_back(output_2.get_node_shared_ptr());
        }

        nms_legacy->set_friendly_name(nms_9->get_friendly_name());
        ngraph::copy_runtime_info(nms_9, new_ops);
        ngraph::replace_node(nms_9, {output_0, nms_legacy->output(1), output_2});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}
