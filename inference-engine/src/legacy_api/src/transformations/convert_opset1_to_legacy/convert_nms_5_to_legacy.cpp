// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph/graph_util.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <legacy/ngraph_ops/nms_ie.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "legacy/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNMS5ToLegacyMatcher, "ConvertNMS5ToLegacyMatcher", 0);

ngraph::pass::ConvertNMS5ToLegacyMatcher::ConvertNMS5ToLegacyMatcher() {
    auto nms = ngraph::pattern::wrap_type<ngraph::opset5::NonMaxSuppression>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto nms_5 = std::dynamic_pointer_cast<ngraph::opset5::NonMaxSuppression>(m.get_match_root());
        if (!nms_5) {
            return false;
        }

        const auto new_args = nms_5->input_values();
        const std::size_t num_of_inputs = new_args.size();

        const auto& arg2 = num_of_inputs > 2 ? new_args.at(2) : ngraph::opset5::Constant::create(element::i32, Shape{}, {0});
        const auto& arg3 = num_of_inputs > 3 ? new_args.at(3) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg4 = num_of_inputs > 4 ? new_args.at(4) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});
        const auto& arg5 = num_of_inputs > 5 ? new_args.at(5) : ngraph::opset5::Constant::create(element::f32, Shape{}, {.0f});

        // vector of new nGraph operations
        NodeVector new_ops;

        auto one_dim_shape = Shape{1};

        auto new_max_per_class = arg2;
        auto new_iou_threshold = arg3;
        auto new_score_threshold = arg4;
        auto new_soft_nms_sigma = arg5;

        new_max_per_class = std::make_shared<ngraph::op::Reshape>(arg2,
                                                                  opset1::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                           one_dim_shape), true);
        new_ops.push_back(new_max_per_class.get_node_shared_ptr());

        new_iou_threshold = std::make_shared<ngraph::op::Reshape>(arg3,
                                                                  opset1::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                           one_dim_shape), true);
        new_ops.push_back(new_iou_threshold.get_node_shared_ptr());

        new_score_threshold = std::make_shared<ngraph::op::Reshape>(arg4,
                                                                    opset1::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                             one_dim_shape), true);
        new_ops.push_back(new_score_threshold.get_node_shared_ptr());

        new_soft_nms_sigma = std::make_shared<ngraph::op::Reshape>(arg5,
                                                                   opset1::Constant::create(ngraph::element::i64, one_dim_shape,
                                                                                            one_dim_shape), true);
        new_ops.push_back(new_soft_nms_sigma.get_node_shared_ptr());

        int center_point_box = 0;
        switch (nms_5->get_box_encoding()) {
            case ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CENTER:
                center_point_box = 1;
                break;
            case ::ngraph::opset5::NonMaxSuppression::BoxEncodingType::CORNER:
                center_point_box = 0;
                break;
            default:
                throw ngraph_error("NonMaxSuppression layer " + nms_5->get_friendly_name() +
                                   " has unsupported box encoding");
        }
        const auto nms_legacy = std::make_shared<op::NonMaxSuppressionIE3>(
                new_args.at(0),
                new_args.at(1),
                new_max_per_class,
                new_iou_threshold,
                new_score_threshold,
                new_soft_nms_sigma,
                center_point_box,
                nms_5->get_sort_result_descending(),
                nms_5->get_output_type());
        new_ops.push_back(nms_legacy);

        nms_legacy->set_friendly_name(nms_5->get_friendly_name());
        ngraph::copy_runtime_info(nms_5, new_ops);
        ngraph::replace_node(nms_5, nms_legacy);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, "ConvertNMS5ToNMSLegacy");
    this->register_matcher(m, callback);
}
