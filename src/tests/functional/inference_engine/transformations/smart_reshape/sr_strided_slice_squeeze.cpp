// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <cpp/ie_cnn_network.h>

#include "common_test_utils/ngraph_test_utils.hpp"

TEST(SmartReshapeTests, SS_Squeeze) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3});
        auto ss = std::make_shared<ngraph::opset5::StridedSlice>(
                input,
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {1, 1}),
                std::vector<int64_t>{1, 1}, std::vector<int64_t>{1, 1});
        auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(ss, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {0}));
        auto relu = std::make_shared<ngraph::opset5::Relu>(squeeze);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);
    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3})) <<
        network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 3}));

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3})) <<
        network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 3}));
}

TEST(SmartReshapeTests, SS_Squeeze_partial_begin_end_mask) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 768});
        auto ss = std::make_shared<ngraph::opset5::StridedSlice>(
                input,
                ngraph::opset5::Constant::create(ngraph::element::i64, {3}, {0, 1, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {3}, {0, 2, 768}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {3}, {1, 1, 1}),
                std::vector<int64_t>{0}, std::vector<int64_t>{1});  // begin_mask.size() is no larger than axis that is going to be squeezed.
        auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(ss, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {1}));
        auto relu = std::make_shared<ngraph::opset5::Relu>(squeeze);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({1, 768})) <<
        network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 128, 768}));

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    auto inputname = network.getFunction()->get_parameters()[0]->get_friendly_name();
    ASSERT_NO_THROW(network.reshape(InferenceEngine::ICNNNetwork::InputShapes{{inputname, {2, 128, 768}}}));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({2, 768})) <<
        network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 128, 768}));
}


TEST(SmartReshapeTests, SS_Squeeze_partial_begin_end) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 768});
        auto ss = std::make_shared<ngraph::opset5::StridedSlice>(
                input,
                ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {0}),  // begin.size() is no larger than axis that is going to be squeezed.
                ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {1}),
                std::vector<int64_t>{1, 1, 1}, std::vector<int64_t>{1, 1, 1});
        auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(ss, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {1}));
        auto relu = std::make_shared<ngraph::opset5::Relu>(squeeze);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({1, 768})) <<
        network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 1, 768}));

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    auto inputname = network.getFunction()->get_parameters()[0]->get_friendly_name();
    ASSERT_NO_THROW(network.reshape(InferenceEngine::ICNNNetwork::InputShapes{{inputname, {2, 1, 768}}}));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({2, 768})) <<
        network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 1, 768}));
}

TEST(SmartReshapeTests, SS_Squeeze_mask_use_negative) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3});
        auto ss = std::make_shared<ngraph::opset5::StridedSlice>(
                input,
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {1, 1}),
                std::vector<int64_t>{1, 1}, std::vector<int64_t>{1, 1}, std::vector<int64_t>{0, 1});
        auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(ss, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {0}));

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{squeeze}, ngraph::ParameterVector{input});
    }


    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({1, 3})) <<
        network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 3}));

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_ANY_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);
}


TEST(SmartReshapeTests, SS_Squeeze_negative_stride_negative) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3});
        auto ss = std::make_shared<ngraph::opset5::StridedSlice>(
                input,
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {-1, -1}),
                std::vector<int64_t>{1, 1}, std::vector<int64_t>{1, 1});
        auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(ss, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {0}));
        auto relu = std::make_shared<ngraph::opset5::Relu>(squeeze);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{input});
    }


    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3})) <<
        network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 3}));

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_ANY_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);
}

TEST(SmartReshapeTests, SS_SharedSqueezes) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3});
        auto ss = std::make_shared<ngraph::opset5::StridedSlice>(
                input,
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {2}, {1, 1}),
                std::vector<int64_t>{1, 1}, std::vector<int64_t>{1, 1});
        auto squeeze_1 = std::make_shared<ngraph::opset5::Squeeze>(ss, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {0}));
        auto squeeze_2 = std::make_shared<ngraph::opset5::Squeeze>(ss, ngraph::opset5::Constant::create(ngraph::element::i64, {1}, {0}));
        auto relu_1 = std::make_shared<ngraph::opset5::Relu>(squeeze_1);
        auto relu_2 = std::make_shared<ngraph::opset5::Relu>(squeeze_2);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu_1, relu_2}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3})) <<
                    network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 3}));

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3})) <<
                    network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 3}));
}


TEST(SmartReshapeTests, SS_SqueezeNegativeAxes) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 1, 8, 1, 2});
        auto ss = std::make_shared<ngraph::opset5::StridedSlice>(
                input,
                ngraph::opset5::Constant::create(ngraph::element::i64, {6}, {0, 0, 0, 0, 0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {6}, {0, 0, 0, 0, 0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {6}, {1, 1, 1, 1, 1, 1}),
                std::vector<int64_t>{1, 1, 1, 1, 1, 1}, std::vector<int64_t>{1, 1, 1, 1, 1, 1});
        auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(ss, ngraph::opset5::Constant::create(ngraph::element::i64, {3}, {-2, 0, -4}));
        auto relu = std::make_shared<ngraph::opset5::Relu>(squeeze);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{relu}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3, 8, 2})) <<
                network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 3, 1, 8, 1, 2}));

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3, 8, 2})) <<
                network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 3, 1, 8, 1, 2}));
}

TEST(SmartReshapeTests, Squeeze_SSNegativeAxes) {
    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        auto input = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 1, 8, 1, 2});
        auto squeeze = std::make_shared<ngraph::opset5::Squeeze>(input, ngraph::opset5::Constant::create(ngraph::element::i64, {3}, {-2, 0, -4}));
        auto ss = std::make_shared<ngraph::opset5::StridedSlice>(
                squeeze,
                ngraph::opset5::Constant::create(ngraph::element::i64, {3}, {0, 0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {3}, {0, 0, 0}),
                ngraph::opset5::Constant::create(ngraph::element::i64, {3}, {1, 1, 1}),
                std::vector<int64_t>{1, 1, 1}, std::vector<int64_t>{1, 1, 1});

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ss}, ngraph::ParameterVector{input});
    }

    InferenceEngine::CNNNetwork network(f);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3, 8, 2})) <<
                network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({1, 3, 1, 8, 1, 2}));

    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_NO_THROW(network.setBatchSize(2));
    check_unique_names(f, unh);

    ASSERT_TRUE(network.getFunction()->get_results()[0]->get_output_partial_shape(0).compatible({3, 8, 2})) <<
                network.getFunction()->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(network.getFunction()->get_parameters()[0]->get_partial_shape().compatible({2, 3, 1, 8, 1, 2}));
}
