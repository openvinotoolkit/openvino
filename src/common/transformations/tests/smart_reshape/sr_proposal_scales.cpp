// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset5.hpp"

using namespace ov;

TEST(SmartReshapeTests, Proposal1Scales) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input_0 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 24, 75, 128});
        auto input_1 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 48, 75, 128});
        auto input_2 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto reshape =
            std::make_shared<opset5::Reshape>(input_2, opset5::Constant::create(element::i64, {1}, {3}), true);
        op::v0::Proposal::Attributes attrs;
        attrs.base_size = 256;
        attrs.box_coordinate_scale = 10.0;
        attrs.box_size_scale = 5.0;
        attrs.clip_after_nms = false;
        attrs.clip_before_nms = true;
        attrs.feat_stride = 8;
        attrs.framework = "tensorflow";
        attrs.min_size = 1;
        attrs.nms_thresh = 0.699999988079f;
        attrs.normalize = true;
        attrs.post_nms_topn = 300;
        attrs.pre_nms_topn = 2147483647;
        attrs.ratio = {0.5, 1.0, 2.0};
        attrs.scale = {0.25, 0.5, 1.0, 2.0};
        auto proposal = std::make_shared<opset1::Proposal>(input_0, input_1, reshape, attrs);
        f = std::make_shared<ov::Model>(NodeVector{proposal}, ParameterVector{input_0, input_1, input_2});
    }

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}

TEST(SmartReshapeTests, Proposal1Scales_WithConvert) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input_0 = std::make_shared<opset5::Parameter>(element::f16, Shape{1, 24, 75, 128});
        auto input_1 = std::make_shared<opset5::Parameter>(element::f16, Shape{1, 48, 75, 128});
        auto input_2 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto input_2_convert = std::make_shared<opset5::Convert>(input_2, element::f16);
        auto reshape =
            std::make_shared<opset5::Reshape>(input_2_convert, opset5::Constant::create(element::i64, {1}, {3}), true);
        op::v0::Proposal::Attributes attrs;
        attrs.base_size = 256;
        attrs.box_coordinate_scale = 10.0;
        attrs.box_size_scale = 5.0;
        attrs.clip_after_nms = false;
        attrs.clip_before_nms = true;
        attrs.feat_stride = 8;
        attrs.framework = "tensorflow";
        attrs.min_size = 1;
        attrs.nms_thresh = 0.699999988079f;
        attrs.normalize = true;
        attrs.post_nms_topn = 300;
        attrs.pre_nms_topn = 2147483647;
        attrs.ratio = {0.5, 1.0, 2.0};
        attrs.scale = {0.25, 0.5, 1.0, 2.0};
        auto proposal = std::make_shared<opset1::Proposal>(input_0, input_1, reshape, attrs);
        f = std::make_shared<ov::Model>(NodeVector{proposal}, ParameterVector{input_0, input_1, input_2});
    }

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}

TEST(SmartReshapeTests, Proposal4Scales) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input_0 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 24, 75, 128});
        auto input_1 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 48, 75, 128});
        auto input_2 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4});
        auto reshape =
            std::make_shared<opset5::Reshape>(input_2, opset5::Constant::create(element::i64, {1}, {-1}), true);
        op::v0::Proposal::Attributes attrs;
        attrs.base_size = 256;
        attrs.box_coordinate_scale = 10.0;
        attrs.box_size_scale = 5.0;
        attrs.clip_after_nms = false;
        attrs.clip_before_nms = true;
        attrs.feat_stride = 8;
        attrs.framework = "tensorflow";
        attrs.min_size = 1;
        attrs.nms_thresh = 0.699999988079f;
        attrs.normalize = true;
        attrs.post_nms_topn = 300;
        attrs.pre_nms_topn = 2147483647;
        attrs.ratio = {0.5, 1.0, 2.0};
        attrs.scale = {0.25, 0.5, 1.0, 2.0};
        auto proposal = std::make_shared<opset5::Proposal>(input_0, input_1, reshape, attrs);
        f = std::make_shared<ov::Model>(NodeVector{proposal}, ParameterVector{input_0, input_1, input_2});
    }

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}

TEST(SmartReshapeTests, Proposal4Scales_WithConvert) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input_0 = std::make_shared<opset5::Parameter>(element::f16, Shape{1, 24, 75, 128});
        auto input_1 = std::make_shared<opset5::Parameter>(element::f16, Shape{1, 48, 75, 128});
        auto input_2 = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 4});
        auto input_2_convert = std::make_shared<opset5::Convert>(input_2, element::f16);
        auto reshape =
            std::make_shared<opset5::Reshape>(input_2_convert, opset5::Constant::create(element::i64, {1}, {-1}), true);
        op::v0::Proposal::Attributes attrs;
        attrs.base_size = 256;
        attrs.box_coordinate_scale = 10.0;
        attrs.box_size_scale = 5.0;
        attrs.clip_after_nms = false;
        attrs.clip_before_nms = true;
        attrs.feat_stride = 8;
        attrs.framework = "tensorflow";
        attrs.min_size = 1;
        attrs.nms_thresh = 0.699999988079f;
        attrs.normalize = true;
        attrs.post_nms_topn = 300;
        attrs.pre_nms_topn = 2147483647;
        attrs.ratio = {0.5, 1.0, 2.0};
        attrs.scale = {0.25, 0.5, 1.0, 2.0};
        auto proposal = std::make_shared<opset5::Proposal>(input_0, input_1, reshape, attrs);
        f = std::make_shared<ov::Model>(NodeVector{proposal}, ParameterVector{input_0, input_1, input_2});
    }

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}
