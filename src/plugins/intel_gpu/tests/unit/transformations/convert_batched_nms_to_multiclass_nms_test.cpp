// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_batched_nms_to_multiclass_nms.hpp"

#include <gtest/gtest.h>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/multiclass_nms.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"

using namespace ov::intel_gpu;

namespace {

std::shared_ptr<ov::Model> make_batched_nms_output_model(int64_t gather_column = 2) {
    auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    auto class_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});

    auto reduce_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1});
    auto reduce_max = std::make_shared<ov::op::v1::ReduceMax>(boxes, reduce_axis, false);
    auto one = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.0f}),
        ov::element::f32);
    auto max_plus_one = std::make_shared<ov::op::v1::Add>(reduce_max, one);
    auto class_ids_convert = std::make_shared<ov::op::v0::Convert>(class_ids, ov::element::f32);
    auto offsets = std::make_shared<ov::op::v1::Multiply>(class_ids_convert, max_plus_one);
    auto offsets_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
        offsets,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto shifted_boxes = std::make_shared<ov::op::v1::Add>(boxes, offsets_unsqueeze);

    auto boxes_shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, -1, 4});
    auto boxes_reshape = std::make_shared<ov::op::v1::Reshape>(shifted_boxes, boxes_shape, false);
    auto scores_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
        scores,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, 1}));

    auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(
        boxes_reshape,
        scores_unsqueeze,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {2000}),
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.5f}),
        std::make_shared<ov::op::v0::Convert>(
            ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.7f}),
            ov::element::f32),
        ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER,
        true,
        ov::element::i64);

    auto gather = std::make_shared<ov::op::v8::Gather>(
        nms,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {gather_column}),
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1}));
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(
        gather,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1}));
    class_ids->get_rt_info()["intel_gpu_batched_nms_static_class_count"] = int64_t{80};
    squeeze->get_rt_info()["intel_gpu_batched_nms_prefix_limit"] = int64_t{100};

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(squeeze)},
        ov::ParameterVector{boxes, scores, class_ids});
}

}  // namespace

TEST(ConvertBatchedNmsToMulticlassNmsTest, ReplacesLoweredBatchedNmsOutputChain) {
    auto model = make_batched_nms_output_model();

    ov::pass::Manager manager;
    manager.register_pass<ConvertBatchedNmsToMulticlassNms>();
    manager.run_passes(model);

    size_t multiclass_nms_count = 0;
    size_t nms_count = 0;
    size_t gather_count = 0;
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::internal::MulticlassNmsIEInternal>(node)) {
            ++multiclass_nms_count;
        }
        if (ov::is_type<ov::op::v9::NonMaxSuppression>(node)) {
            ++nms_count;
        }
        if (ov::is_type<ov::op::v8::Gather>(node)) {
            ++gather_count;
        }
    }

    EXPECT_EQ(multiclass_nms_count, 1);
    EXPECT_EQ(nms_count, 0);
    EXPECT_EQ(gather_count, 1);

    auto result_input = model->get_results().front()->input_value(0).get_node_shared_ptr();
    auto selected_box_indices = ov::as_type_ptr<ov::op::v8::Gather>(result_input);
    ASSERT_NE(selected_box_indices, nullptr);

    auto valid_selected_indices =
        ov::as_type_ptr<ov::op::v8::Slice>(selected_box_indices->input_value(0).get_node_shared_ptr());
    ASSERT_NE(valid_selected_indices, nullptr);

    auto multiclass_nms = ov::as_type_ptr<ov::op::internal::MulticlassNmsIEInternal>(
        valid_selected_indices->input_value(0).get_node_shared_ptr());
    ASSERT_NE(multiclass_nms, nullptr);
    EXPECT_EQ(multiclass_nms->get_attrs().sort_result_type,
              ov::op::util::MulticlassNmsBase::SortResultType::SCORE);
    EXPECT_EQ(multiclass_nms->get_attrs().output_type, ov::element::i64);
    EXPECT_EQ(multiclass_nms->get_attrs().nms_top_k, 2000);
    EXPECT_EQ(multiclass_nms->get_attrs().keep_top_k, 100);
    EXPECT_EQ(multiclass_nms->get_input_element_type(0), ov::element::f32);
    EXPECT_EQ(multiclass_nms->get_input_element_type(1), ov::element::f32);
    EXPECT_EQ(valid_selected_indices->input_value(2), multiclass_nms->output(2));
}

TEST(ConvertBatchedNmsToMulticlassNmsTest, KeepsGenericNmsGatherChainUntouched) {
    auto model = make_batched_nms_output_model(1);

    ov::pass::Manager manager;
    manager.register_pass<ConvertBatchedNmsToMulticlassNms>();
    manager.run_passes(model);

    size_t multiclass_nms_count = 0;
    size_t nms_count = 0;
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::internal::MulticlassNmsIEInternal>(node)) {
            ++multiclass_nms_count;
        }
        if (ov::is_type<ov::op::v9::NonMaxSuppression>(node)) {
            ++nms_count;
        }
    }

    EXPECT_EQ(multiclass_nms_count, 0);
    EXPECT_EQ(nms_count, 1);
}

