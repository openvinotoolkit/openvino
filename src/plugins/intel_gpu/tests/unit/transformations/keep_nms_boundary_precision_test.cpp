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

struct NmsTestModel {
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::Node> boxes_offset_add;
    std::shared_ptr<ov::Node> offsets_unsqueeze;
    std::shared_ptr<ov::Node> offsets_multiply;
    std::shared_ptr<ov::Node> class_ids_convert;
    std::shared_ptr<ov::Node> max_plus_one;
    std::shared_ptr<ov::Node> reduce_max;
    std::shared_ptr<ov::Node> max_output_boxes;
    std::shared_ptr<ov::Node> iou_threshold;
    std::shared_ptr<ov::Node> score_threshold;
    std::shared_ptr<ov::Node> boxes_reshape;
    std::shared_ptr<ov::Node> scores_unsqueeze;
    std::shared_ptr<ov::Node> nms;
};

NmsTestModel make_nms_model(const std::string& nms_prefix, bool use_batched_nms_offsets = true) {
    auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    boxes->set_friendly_name("boxes");
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    scores->set_friendly_name("scores");
    auto class_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    class_ids->set_friendly_name("class_ids");

    auto reduce_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1});
    auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(boxes, reduce_axis, false);
    reduce_max->set_friendly_name("aten::max/ReduceMax");

    auto one_compressed = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f});
    one_compressed->set_friendly_name("aten::add/Multiply_3_compressed");
    auto one = std::make_shared<ov::op::v0::Convert>(one_compressed, ov::element::f32);
    one->set_friendly_name("aten::add/Multiply_3");
    auto max_plus_one = std::make_shared<ov::op::v1::Add>(reduce_max, one);
    max_plus_one->set_friendly_name("aten::add/Add_3");

    std::shared_ptr<ov::Node> offsets_input = scores;
    std::shared_ptr<ov::Node> class_ids_convert;
    if (use_batched_nms_offsets) {
        class_ids_convert = std::make_shared<ov::op::v0::Convert>(class_ids, ov::element::f32);
        class_ids_convert->set_friendly_name("aten::to/ConvertLike_2");
        offsets_input = class_ids_convert;
    }

    auto offsets = std::make_shared<ov::op::v1::Multiply>(offsets_input, max_plus_one);
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

    return NmsTestModel{std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{boxes, scores, class_ids}),
                        boxes_for_nms,
                        offsets_unsqueeze,
                        offsets,
                        class_ids_convert,
                        max_plus_one,
                        reduce_max,
                        max_output_boxes,
                        iou_threshold,
                        score_threshold,
                        boxes_reshape,
                        scores_unsqueeze,
                        nms};
}

}  // namespace

TEST(KeepNMSBoundaryPrecisionTest, MarksTargetNmsSubgraph) {
    auto test_model = make_nms_model("torchvision::nms/");

    ov::pass::Manager manager;
    manager.register_pass<KeepNMSBoundaryPrecision>();
    manager.run_passes(test_model.model);

    for (const auto& node : {test_model.nms,
                             test_model.boxes_reshape,
                             test_model.scores_unsqueeze,
                             test_model.boxes_offset_add,
                             test_model.offsets_unsqueeze,
                             test_model.offsets_multiply,
                             test_model.class_ids_convert,
                             test_model.max_plus_one,
                             test_model.reduce_max,
                             test_model.max_output_boxes,
                             test_model.iou_threshold,
                             test_model.score_threshold}) {
        ASSERT_NE(node, nullptr);
        EXPECT_TRUE(ov::is_conversion_disabled(node, ov::element::f16));
    }
}

TEST(KeepNMSBoundaryPrecisionTest, MarksStructuralPatternWithoutTorchvisionNames) {
    auto test_model = make_nms_model("custom::nms/");

    ov::pass::Manager manager;
    manager.register_pass<KeepNMSBoundaryPrecision>();
    manager.run_passes(test_model.model);

    ASSERT_NE(test_model.nms, nullptr);
    EXPECT_TRUE(ov::is_conversion_disabled(test_model.nms, ov::element::f16));
}

TEST(KeepNMSBoundaryPrecisionTest, IgnoresNmsWithoutBatchedOffsetChain) {
    auto test_model = make_nms_model("custom::nms/", false);

    ov::pass::Manager manager;
    manager.register_pass<KeepNMSBoundaryPrecision>();
    manager.run_passes(test_model.model);

    ASSERT_NE(test_model.nms, nullptr);
    EXPECT_FALSE(ov::is_conversion_disabled(test_model.nms, ov::element::f16));
    EXPECT_EQ(test_model.class_ids_convert, nullptr);
}