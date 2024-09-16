// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_nms_rotated_to_nms_ie_internal.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/nms_rotated.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/nms_ie_internal.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertNMSRotatedToNMSIEInternal::ConvertNMSRotatedToNMSIEInternal() {
    MATCHER_SCOPE(ConvertNMSRotatedToNMSIEInternal);
    auto nms = ov::pass::pattern::wrap_type<ov::op::v13::NMSRotated>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto nms_rotated = ov::as_type_ptr<ov::op::v13::NMSRotated>(m.get_match_root());
        if (!nms_rotated || transformation_callback(nms_rotated)) {
            return false;
        }

        const auto new_args = nms_rotated->input_values();
        const std::size_t num_of_inputs = new_args.size();
        OPENVINO_ASSERT(num_of_inputs == 5);

        const auto& max_per_class = new_args.at(2);
        const auto& iou_threshold = new_args.at(3);
        const auto& score_threshold = new_args.at(4);

        // vector of new openvino operations
        NodeVector new_ops;

        auto one_dim_shape = Shape{1};

        Output<Node> new_max_per_class;
        Output<Node> new_iou_threshold;
        Output<Node> new_score_threshold;
        Output<Node> new_soft_nms_sigma;

        Output<Node> new_shape_for_max_per_class = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {1});
        Output<Node> new_shape_for_iou_threshold = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {1});
        Output<Node> new_shape_for_score_threshold = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {1});
        Output<Node> new_shape_for_soft_nms_sigma = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {1});

        new_max_per_class = std::make_shared<ov::op::v1::Reshape>(max_per_class, new_shape_for_max_per_class, true);
        new_ops.emplace_back(new_max_per_class.get_node_shared_ptr());

        new_iou_threshold = std::make_shared<ov::op::v1::Reshape>(iou_threshold, new_shape_for_iou_threshold, true);
        new_ops.emplace_back(new_iou_threshold.get_node_shared_ptr());

        new_score_threshold =
            std::make_shared<ov::op::v1::Reshape>(score_threshold, new_shape_for_score_threshold, true);
        new_ops.emplace_back(new_score_threshold.get_node_shared_ptr());

        constexpr int BoxEncodingType_Center = 1;             // see NonMaxSuppression::BoxEncodingType
        const int center_point_box = BoxEncodingType_Center;  // for NMSRotated is it always Center

        const auto rotation = nms_rotated->get_clockwise()
                                  ? op::internal::NonMaxSuppressionIEInternal::Rotation_Clockwise
                                  : op::internal::NonMaxSuppressionIEInternal::Rotation_Counterclockwise;

        std::shared_ptr<op::internal::NonMaxSuppressionIEInternal> nms_legacy{nullptr};

        nms_legacy =
            std::make_shared<op::internal::NonMaxSuppressionIEInternal>(new_args.at(0),
                                                                        new_args.at(1),

                                                                        new_max_per_class,
                                                                        new_iou_threshold,
                                                                        new_score_threshold,

                                                                        center_point_box,
                                                                        nms_rotated->get_sort_result_descending(),
                                                                        element::i32,
                                                                        nms_rotated->get_output_element_type(1),
                                                                        rotation);
        new_ops.push_back(nms_legacy);

        Output<Node> output_0 = nms_legacy->output(0);
        if (nms_rotated->output(0).get_element_type() != output_0.get_element_type()) {
            output_0 = std::make_shared<ov::op::v0::Convert>(output_0, nms_rotated->output(0).get_element_type());
            new_ops.emplace_back(output_0.get_node_shared_ptr());
        }

        Output<Node> output_2 = nms_legacy->output(2);
        if (nms_rotated->output(2).get_element_type() != output_2.get_element_type()) {
            output_2 = std::make_shared<ov::op::v0::Convert>(output_2, nms_rotated->output(2).get_element_type());
            new_ops.emplace_back(output_2.get_node_shared_ptr());
        }

        nms_legacy->set_friendly_name(nms_rotated->get_friendly_name());
        ov::copy_runtime_info(nms_rotated, new_ops);
        ov::replace_node(nms_rotated, {output_0, nms_legacy->output(1), output_2});
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}
