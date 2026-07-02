// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/keep_nms_boundary_precision.hpp"

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace ov::intel_gpu;

namespace {

std::shared_ptr<ov::Model> make_nms_model(const std::string& nms_prefix) {
    auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    boxes->set_friendly_name("boxes");
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    scores->set_friendly_name("scores");

    auto reduce_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
    auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(boxes, reduce_axis, false);
    reduce_max->set_friendly_name("aten::max/ReduceMax");

    auto one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto max_plus_one = std::make_shared<ov::op::v1::Add>(reduce_max, one);
    max_plus_one->set_friendly_name("aten::add/Add_3");

    auto offsets = std::make_shared<ov::op::v1::Multiply>(scores, max_plus_one);
    offsets->set_friendly_name("aten::mul/Multiply_2");

    auto unsqueeze_axis_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto offsets_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(offsets, unsqueeze_axis_1);
    offsets_unsqueeze->set_friendly_name("aten::unsqueeze/Unsqueeze_2");

    auto boxes_for_nms = std::make_shared<ov::op::v1::Add>(boxes, offsets_unsqueeze);
    boxes_for_nms->set_friendly_name("aten::add/Add_4");

    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, -1, 4});
    auto boxes_reshape = std::make_shared<ov::op::v1::Reshape>(boxes_for_nms, reshape_shape, false);
    boxes_reshape->set_friendly_name(nms_prefix + "Reshape");

    auto score_unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1});
    auto scores_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(scores, score_unsqueeze_axis);
    scores_unsqueeze->set_friendly_name(nms_prefix + "Unsqueeze");

    auto max_output_boxes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2000});
    max_output_boxes->set_friendly_name(nms_prefix + "Constant_max_output_boxes");
    auto iou_threshold = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5f});
    iou_threshold->set_friendly_name(nms_prefix + "Constant_iou");
    auto score_threshold_compressed = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.7f});
    score_threshold_compressed->set_friendly_name("2636_1_compressed");
    auto score_threshold = std::make_shared<ov::op::v0::Convert>(score_threshold_compressed, ov::element::f32);
    score_threshold->set_friendly_name("2636_1");

    auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(boxes_reshape,
                                                               scores_unsqueeze,
                                                               max_output_boxes,
                                                               iou_threshold,
                                                               score_threshold,
                                                               ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER,
                                                               true,
                                                               ov::element::i64);
    nms->set_friendly_name(nms_prefix + "NonMaxSuppression");

    auto result = std::make_shared<ov::op::v0::Result>(nms->output(0));
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{boxes, scores});
}

std::shared_ptr<ov::Node> find_node_by_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == name) {
            return node;
        }
    }
    return nullptr;
}

}  // namespace

TEST(KeepNMSBoundaryPrecisionTest, MarksTargetNmsSubgraph) {
    auto model = make_nms_model("torchvision::nms/");

    ov::pass::Manager manager;
    manager.register_pass<KeepNMSBoundaryPrecision>();
    manager.run_passes(model);

    for (const auto& name : {std::string{"torchvision::nms/NonMaxSuppression"},
                             std::string{"torchvision::nms/Reshape"},
                             std::string{"torchvision::nms/Unsqueeze"},
                             std::string{"aten::add/Add_4"},
                             std::string{"aten::unsqueeze/Unsqueeze_2"},
                             std::string{"aten::mul/Multiply_2"},
                             std::string{"aten::add/Add_3"},
                             std::string{"aten::max/ReduceMax"},
                             std::string{"torchvision::nms/Constant_max_output_boxes"},
                             std::string{"torchvision::nms/Constant_iou"},
                             std::string{"2636_1"}}) {
        auto node = find_node_by_name(model, name);
        ASSERT_NE(node, nullptr) << name;
        EXPECT_TRUE(ov::is_conversion_disabled(node, ov::element::f16)) << name;
    }
}

TEST(KeepNMSBoundaryPrecisionTest, IgnoresOtherNmsNames) {
    auto model = make_nms_model("custom::nms/");

    ov::pass::Manager manager;
    manager.register_pass<KeepNMSBoundaryPrecision>();
    manager.run_passes(model);

    auto node = find_node_by_name(model, "custom::nms/NonMaxSuppression");
    ASSERT_NE(node, nullptr);
    EXPECT_FALSE(ov::is_conversion_disabled(node, ov::element::f16));
}