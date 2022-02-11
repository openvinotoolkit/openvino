// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <cpp/ie_cnn_network.h>

#include "common_test_utils/ngraph_test_utils.hpp"

TEST(SmartReshapeTests, Proposal1Scales) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input_0 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 24, 75, 128});
        auto input_1 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 48, 75, 128});
        auto input_2 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3});
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input_2, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {3}), true);
        ngraph::op::ProposalAttrs attrs;
        attrs.base_size = 256;
        attrs.box_coordinate_scale = 10.0;
        attrs.box_size_scale = 5.0;
        attrs.clip_after_nms = false;
        attrs.clip_before_nms = true;
        attrs.feat_stride = 8;
        attrs.framework = "tensorflow";
        attrs.min_size = 1;
        attrs.nms_thresh = 0.699999988079;
        attrs.normalize = true;
        attrs.post_nms_topn = 300;
        attrs.pre_nms_topn = 2147483647;
        attrs.ratio = {0.5, 1.0, 2.0};
        attrs.scale = {0.25, 0.5, 1.0, 2.0};
        auto proposal = std::make_shared<ngraph::opset1::Proposal>(input_0, input_1, reshape, attrs);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{proposal}, ngraph::ParameterVector{input_0, input_1, input_2});
    }

    InferenceEngine::CNNNetwork network(f);
    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);
    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({600, 5}));
}

TEST(SmartReshapeTests, Proposal1Scales_WithConvert) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input_0 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f16, ngraph::Shape{1, 24, 75, 128});
        auto input_1 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f16, ngraph::Shape{1, 48, 75, 128});
        auto input_2 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3});
        auto input_2_convert = std::make_shared<ngraph::opset5::Convert>(input_2, ngraph::element::f16);
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input_2_convert,
                                                                 ngraph::opset5::Constant::create(
                                                                         ngraph::element::i64, {1}, {3}), true);
        ngraph::op::ProposalAttrs attrs;
        attrs.base_size = 256;
        attrs.box_coordinate_scale = 10.0;
        attrs.box_size_scale = 5.0;
        attrs.clip_after_nms = false;
        attrs.clip_before_nms = true;
        attrs.feat_stride = 8;
        attrs.framework = "tensorflow";
        attrs.min_size = 1;
        attrs.nms_thresh = 0.699999988079;
        attrs.normalize = true;
        attrs.post_nms_topn = 300;
        attrs.pre_nms_topn = 2147483647;
        attrs.ratio = {0.5, 1.0, 2.0};
        attrs.scale = {0.25, 0.5, 1.0, 2.0};
        auto proposal = std::make_shared<ngraph::opset1::Proposal>(input_0, input_1, reshape, attrs);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{proposal}, ngraph::ParameterVector{input_0, input_1, input_2});
    }

    InferenceEngine::CNNNetwork network(f);
    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);
    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({600, 5}));
}

TEST(SmartReshapeTests, Proposal4Scales) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input_0 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 24, 75, 128});
        auto input_1 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 48, 75, 128});
        auto input_2 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 4});
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input_2,
                                                                 ngraph::opset5::Constant::create(
                                                                         ngraph::element::i64, {1}, {-1}), true);
        ngraph::op::ProposalAttrs attrs;
        attrs.base_size = 256;
        attrs.box_coordinate_scale = 10.0;
        attrs.box_size_scale = 5.0;
        attrs.clip_after_nms = false;
        attrs.clip_before_nms = true;
        attrs.feat_stride = 8;
        attrs.framework = "tensorflow";
        attrs.min_size = 1;
        attrs.nms_thresh = 0.699999988079;
        attrs.normalize = true;
        attrs.post_nms_topn = 300;
        attrs.pre_nms_topn = 2147483647;
        attrs.ratio = {0.5, 1.0, 2.0};
        attrs.scale = {0.25, 0.5, 1.0, 2.0};
        auto proposal = std::make_shared<ngraph::opset5::Proposal>(input_0, input_1, reshape, attrs);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{proposal}, ngraph::ParameterVector{input_0, input_1, input_2});
    }

    InferenceEngine::CNNNetwork network(f);

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({600, 5}));
}

TEST(SmartReshapeTests, Proposal4Scales_WithConvert) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input_0 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f16, ngraph::Shape{1, 24, 75, 128});
        auto input_1 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f16, ngraph::Shape{1, 48, 75, 128});
        auto input_2 = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 4});
        auto input_2_convert = std::make_shared<ngraph::opset5::Convert>(input_2, ngraph::element::f16);
        auto reshape = std::make_shared<ngraph::opset5::Reshape>(input_2_convert,
                                                                 ngraph::opset5::Constant::create(
                                                                         ngraph::element::i64, {1}, {-1}), true);
        ngraph::op::ProposalAttrs attrs;
        attrs.base_size = 256;
        attrs.box_coordinate_scale = 10.0;
        attrs.box_size_scale = 5.0;
        attrs.clip_after_nms = false;
        attrs.clip_before_nms = true;
        attrs.feat_stride = 8;
        attrs.framework = "tensorflow";
        attrs.min_size = 1;
        attrs.nms_thresh = 0.699999988079;
        attrs.normalize = true;
        attrs.post_nms_topn = 300;
        attrs.pre_nms_topn = 2147483647;
        attrs.ratio = {0.5, 1.0, 2.0};
        attrs.scale = {0.25, 0.5, 1.0, 2.0};
        auto proposal = std::make_shared<ngraph::opset5::Proposal>(input_0, input_1, reshape, attrs);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{proposal},
                                               ngraph::ParameterVector{input_0, input_1, input_2});
    }

    InferenceEngine::CNNNetwork network(f);

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({600, 5}));
}