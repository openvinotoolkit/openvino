// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"

using namespace ov;

TEST(SmartReshapeTests, SS_Squeeze) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto ss = std::make_shared<opset5::StridedSlice>(input,
                                                         opset5::Constant::create(element::i64, {2}, {0, 0}),
                                                         opset5::Constant::create(element::i64, {2}, {0, 0}),
                                                         opset5::Constant::create(element::i64, {2}, {1, 1}),
                                                         std::vector<int64_t>{1, 1},
                                                         std::vector<int64_t>{1, 1});
        auto squeeze = std::make_shared<opset5::Squeeze>(ss, opset5::Constant::create(element::i64, {1}, {0}));
        auto relu = std::make_shared<opset5::Relu>(squeeze);

        f = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({3}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 3}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}

TEST(SmartReshapeTests, SS_Squeeze_partial_begin_end_mask) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128, 768});
        auto ss = std::make_shared<opset5::StridedSlice>(
            input,
            opset5::Constant::create(element::i64, {3}, {0, 1, 0}),
            opset5::Constant::create(element::i64, {3}, {0, 2, 768}),
            opset5::Constant::create(element::i64, {3}, {1, 1, 1}),
            std::vector<int64_t>{0},
            std::vector<int64_t>{1});  // begin_mask.size() is no larger than axis that is going to be squeezed.
        auto squeeze = std::make_shared<opset5::Squeeze>(ss, opset5::Constant::create(element::i64, {1}, {1}));
        auto relu = std::make_shared<opset5::Relu>(squeeze);

        f = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({1, 768}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 128, 768}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    auto inputname = f->get_parameters()[0]->get_friendly_name();
    OV_ASSERT_NO_THROW(f->reshape({{2, 128, 768}}));
    check_unique_names(f, unh);

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({2, 768}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({2, 128, 768}));
}

TEST(SmartReshapeTests, SS_Squeeze_partial_begin_end) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 768});
        auto ss = std::make_shared<opset5::StridedSlice>(
            input,
            opset5::Constant::create(element::i64,
                                     {1},
                                     {0}),  // begin.size() is no larger than axis that is going to be squeezed.
            opset5::Constant::create(element::i64, {1}, {0}),
            opset5::Constant::create(element::i64, {1}, {1}),
            std::vector<int64_t>{1, 1, 1},
            std::vector<int64_t>{1, 1, 1});
        auto squeeze = std::make_shared<opset5::Squeeze>(ss, opset5::Constant::create(element::i64, {1}, {1}));
        auto relu = std::make_shared<opset5::Relu>(squeeze);

        f = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({1, 768}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 1, 768}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    auto inputname = f->get_parameters()[0]->get_friendly_name();
    OV_ASSERT_NO_THROW(f->reshape({{2, 1, 768}}));
    check_unique_names(f, unh);

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({2, 768}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({2, 1, 768}));
}

TEST(SmartReshapeTests, SS_Squeeze_mask_use_negative) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto ss = std::make_shared<opset5::StridedSlice>(input,
                                                         opset5::Constant::create(element::i64, {2}, {0, 0}),
                                                         opset5::Constant::create(element::i64, {2}, {0, 0}),
                                                         opset5::Constant::create(element::i64, {2}, {1, 1}),
                                                         std::vector<int64_t>{1, 1},
                                                         std::vector<int64_t>{1, 1},
                                                         std::vector<int64_t>{0, 1});
        auto squeeze = std::make_shared<opset5::Squeeze>(ss, opset5::Constant::create(element::i64, {1}, {0}));

        f = std::make_shared<ov::Model>(NodeVector{squeeze}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({1, 3}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 3}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_ANY_THROW(set_batch(f, 2));
    check_unique_names(f, unh);
}

TEST(SmartReshapeTests, SS_Squeeze_negative_stride_negative) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto ss = std::make_shared<opset5::StridedSlice>(input,
                                                         opset5::Constant::create(element::i64, {2}, {0, 0}),
                                                         opset5::Constant::create(element::i64, {2}, {0, 0}),
                                                         opset5::Constant::create(element::i64, {2}, {-1, -1}),
                                                         std::vector<int64_t>{1, 1},
                                                         std::vector<int64_t>{1, 1});
        auto squeeze = std::make_shared<opset5::Squeeze>(ss, opset5::Constant::create(element::i64, {1}, {0}));
        auto relu = std::make_shared<opset5::Relu>(squeeze);

        f = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({3}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 3}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    ASSERT_ANY_THROW(set_batch(f, 2));
    check_unique_names(f, unh);
}

TEST(SmartReshapeTests, SS_SharedSqueezes) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3});
        auto ss = std::make_shared<opset5::StridedSlice>(input,
                                                         opset5::Constant::create(element::i64, {2}, {0, 0}),
                                                         opset5::Constant::create(element::i64, {2}, {0, 0}),
                                                         opset5::Constant::create(element::i64, {2}, {1, 1}),
                                                         std::vector<int64_t>{1, 1},
                                                         std::vector<int64_t>{1, 1});
        auto squeeze_1 = std::make_shared<opset5::Squeeze>(ss, opset5::Constant::create(element::i64, {1}, {0}));
        auto squeeze_2 = std::make_shared<opset5::Squeeze>(ss, opset5::Constant::create(element::i64, {1}, {0}));
        auto relu_1 = std::make_shared<opset5::Relu>(squeeze_1);
        auto relu_2 = std::make_shared<opset5::Relu>(squeeze_2);
        f = std::make_shared<ov::Model>(NodeVector{relu_1, relu_2}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({3}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 3}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}

TEST(SmartReshapeTests, SS_SqueezeNegativeAxes) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 1, 8, 1, 2});
        auto ss =
            std::make_shared<opset5::StridedSlice>(input,
                                                   opset5::Constant::create(element::i64, {6}, {0, 0, 0, 0, 0, 0}),
                                                   opset5::Constant::create(element::i64, {6}, {0, 0, 0, 0, 0, 0}),
                                                   opset5::Constant::create(element::i64, {6}, {1, 1, 1, 1, 1, 1}),
                                                   std::vector<int64_t>{1, 1, 1, 1, 1, 1},
                                                   std::vector<int64_t>{1, 1, 1, 1, 1, 1});
        auto squeeze = std::make_shared<opset5::Squeeze>(ss, opset5::Constant::create(element::i64, {3}, {-2, 0, -4}));
        auto relu = std::make_shared<opset5::Relu>(squeeze);

        f = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({3, 8, 2}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 3, 1, 8, 1, 2}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}

TEST(SmartReshapeTests, Squeeze_SSNegativeAxes) {
    std::shared_ptr<ov::Model> f(nullptr);
    {
        auto input = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 3, 1, 8, 1, 2});
        auto squeeze =
            std::make_shared<opset5::Squeeze>(input, opset5::Constant::create(element::i64, {3}, {-2, 0, -4}));
        auto ss = std::make_shared<opset5::StridedSlice>(squeeze,
                                                         opset5::Constant::create(element::i64, {3}, {0, 0, 0}),
                                                         opset5::Constant::create(element::i64, {3}, {0, 0, 0}),
                                                         opset5::Constant::create(element::i64, {3}, {1, 1, 1}),
                                                         std::vector<int64_t>{1, 1, 1},
                                                         std::vector<int64_t>{1, 1, 1});

        f = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{input});
    }

    ASSERT_TRUE(f->get_results()[0]->get_output_partial_shape(0).compatible({3, 8, 2}))
        << f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(f->get_parameters()[0]->get_partial_shape().compatible({1, 3, 1, 8, 1, 2}));

    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    init_unique_names(f, unh);
    EXPECT_ANY_THROW(set_batch(f, 2));
}
