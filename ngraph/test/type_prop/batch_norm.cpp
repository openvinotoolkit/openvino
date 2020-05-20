//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, batch_norm_training_rank_less_than_2)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{1});
    try
    {
        auto bc = make_shared<op::BatchNormTraining>(dummy, dummy, dummy, 0.001);
        FAIL() << "BatchNorm c-tor should throw for tensors whose rank is less than 2";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input argument must have rank of at least 2"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_zero_channel_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{1, 0, 2, 3});
    auto gamma = make_shared<op::Parameter>(element::f32, Shape{0});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{0});
    try
    {
        auto bc = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, 0.001);
        FAIL() << "BatchNorm c-tor should throw for tensors w/ zero-dimension channels";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_et_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto gamma = make_shared<op::Parameter>(element::f64, Shape{3});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{3});

    try
    {
        auto bc = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, 0.001);
        FAIL() << "BatchNorm c-tor should throw for different element types";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_shape_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto gamma = make_shared<op::Parameter>(element::f32, Shape{4});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{3});

    try
    {
        auto bc = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, 0.001);
        FAIL() << "BatchNorm c-tor should throw if gamma and beta shapes don't match";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Shapes for gamma/beta do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_backprop_et_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto gamma = make_shared<op::Parameter>(element::f32, Shape{3});
    auto beta = make_shared<op::Parameter>(element::f64, Shape{3});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{3});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{3});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});

    try
    {
        auto bc = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, 0.001);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input element types do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_backprop_shape_check)
{
    auto data_batch = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto gamma = make_shared<op::Parameter>(element::f32, Shape{3});
    auto beta = make_shared<op::Parameter>(element::f32, Shape{4});
    auto mean = make_shared<op::Parameter>(element::f32, Shape{3});
    auto variance = make_shared<op::Parameter>(element::f32, Shape{3});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});

    try
    {
        auto bc = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, 0.001);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_backprop_delta_check)
{
    auto dummy = make_shared<op::Parameter>(element::f32, Shape{3});
    auto dummy2 = make_shared<op::Parameter>(element::f32, Shape{4});
    auto param = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 2});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{4, 3, 2, 3});

    try
    {
        auto bc = make_shared<op::BatchNormTrainingBackprop>(
            param, dummy, dummy, dummy, dummy, delta, 0.001);
        FAIL() << "Deduced type should disagree with c-tor arguments";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Shape of delta does not match the shape of the input data"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_inference_partial_all_rank_dynamic)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn = make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, batch_norm_inference_partial_input_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn = make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, batch_norm_inference_partial_input_rank_static_dynamic_zero_channels)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);
        FAIL() << "Zero channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn = make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, batch_norm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_wrong_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);
        FAIL() << "Wrong gamma/beta/mean/variance shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape for gamma/beta/mean/variance ({?,?}) does not have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batch_norm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_inconsistent_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3, Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);
        FAIL() << "Inconsistent gamma/beta/mean/variance shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batch_norm_inference_partial_input_rank_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{4};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);
        FAIL() << "Inconsistent gamma/beta/mean/variance channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_inference_partial_input_rank_static_dynamic_some_static_ok)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn = make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 3, Dimension::dynamic(), 224}));
}

TEST(type_prop,
     batch_norm_inference_partial_input_rank_static_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{64, 4, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn =
            make_shared<op::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);
        FAIL() << "Inconsistent input/gamma/beta/mean/variance channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input channel dimension (4) does not match "
                                         "shape for gamma/beta/mean/variance ({3})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_partial_all_rank_dynamic)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batch_norm_training_partial_input_rank_static_dynamic_batch_size_known_ok)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batch_norm_training_partial_input_rank_static_dynamic_channel_count_known_ok)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape{3}));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape{3}));
}

TEST(type_prop, batch_norm_training_partial_input_rank_static_dynamic_zero_channels)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    try
    {
        auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);
        FAIL() << "Zero channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_partial_input_rank_dynamic_some_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batch_norm_training_partial_input_rank_dynamic_some_rank_static_dynamic_wrong_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);
        FAIL() << "Wrong gamma/beta shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shape for gamma/beta ({?,?}) does not have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batch_norm_training_partial_input_rank_dynamic_some_rank_static_dynamic_inconsistent_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3, Dimension::dynamic()};
    PartialShape beta_shape{Dimension::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);
        FAIL() << "Inconsistent gamma/beta shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Shapes for gamma/beta do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batch_norm_training_partial_input_rank_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{4};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);
        FAIL() << "Inconsistent gamma/beta channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Shapes for gamma/beta do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_partial_input_rank_static_dynamic_some_static_ok)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{3};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 3, Dimension::dynamic(), 224}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape{3}));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape{3}));
}

TEST(type_prop,
     batch_norm_training_partial_input_rank_static_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{64, 4, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTraining>(data_batch, gamma, beta, epsilon);
        FAIL() << "Inconsistent input/gamma/beta channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Input channel dimension (4) does not match shape for gamma/beta ({3})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

////
////
////
////

TEST(type_prop, batch_norm_training_backprop_partial_all_rank_dynamic)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        data_batch, gamma, beta, mean, variance, delta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batch_norm_training_backprop_partial_input_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        data_batch, gamma, beta, mean, variance, delta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batch_norm_training_backprop_partial_input_rank_static_dynamic_zero_channels)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, epsilon);
        FAIL() << "Zero channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, batch_norm_training_backprop_partial_delta_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        data_batch, gamma, beta, mean, variance, delta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(type_prop, batch_norm_training_backprop_partial_delta_rank_static_dynamic_channels_known)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 5, Dimension::dynamic(), Dimension::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        data_batch, gamma, beta, mean, variance, delta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{Dimension::dynamic(), 5, Dimension::dynamic(), Dimension::dynamic()}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape{5}));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape{5}));
}

TEST(type_prop, batch_norm_training_backprop_partial_delta_rank_static_dynamic_zero_channels)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, epsilon);
        FAIL() << "Zero channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Channel count must be at least 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batch_norm_training_backprop_partial_input_and_delta_rank_dynamic_some_rank_static_dynamic_ok)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        data_batch, gamma, beta, mean, variance, delta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape::dynamic(1)));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape::dynamic(1)));
}

TEST(
    type_prop,
    batch_norm_training_backprop_partial_input_and_delta_rank_dynamic_some_rank_static_dynamic_wrong_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, epsilon);
        FAIL() << "Wrong gamma/beta/mean/variance shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape for gamma/beta/mean/variance ({?,?}) does not have rank 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    batch_norm_training_backprop_partial_input_and_delta_rank_dynamic_some_rank_static_dynamic_inconsistent_rank)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3, Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, epsilon);
        FAIL() << "Wrong gamma/beta/mean/variance shape not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    batch_norm_training_backprop_partial_input_and_delta_rank_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{4};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, epsilon);
        FAIL() << "nconsistent gamma/beta/mean/variance channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Shapes for gamma/beta/mean/variance do not match"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop,
     batch_norm_training_backprop_partial_input_and_delta_rank_static_dynamic_some_static_ok)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 3, 448, 224};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    auto bn = make_shared<op::BatchNormTrainingBackprop>(
        data_batch, gamma, beta, mean, variance, delta, epsilon);

    ASSERT_EQ(bn->get_output_size(), 3);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(1), data_batch_et);
    ASSERT_EQ(bn->get_output_element_type(2), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(PartialShape{64, 3, 448, 224}));
    ASSERT_TRUE(bn->get_output_partial_shape(1).same_scheme(PartialShape{3}));
    ASSERT_TRUE(bn->get_output_partial_shape(2).same_scheme(PartialShape{3}));
}

TEST(
    type_prop,
    batch_norm_training_backprop_partial_input_and_delta_rank_static_dynamic_some_static_inconsistent_channel_count)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 4, 448, 224};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, epsilon);
        FAIL() << "Inconsistent delta/gamma/beta/mean/variance channel count not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input channel dimension (4) does not match "
                                         "shape for gamma/beta/mean/variance ({3})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    batch_norm_training_backprop_partial_input_and_delta_rank_static_dynamic_some_static_inconsistent_batch_size)
{
    PartialShape data_batch_shape{64, 3, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{128, 4, Dimension::dynamic(), 224};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, epsilon);
        FAIL() << "Inconsistent input/delta batch size not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape of delta does not match the shape of the input data (input data "
                        "shape: {64,3,?,224}, delta shape: {128,4,?,224})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(
    type_prop,
    batch_norm_training_backprop_partial_input_and_delta_rank_static_dynamic_some_static_inconsistent_spatial_dims)
{
    PartialShape data_batch_shape{Dimension::dynamic(), 3, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    PartialShape delta_shape{Dimension::dynamic(), 3, Dimension::dynamic(), 448};
    double epsilon = 0.001;
    element::Type data_batch_et = element::f32;
    element::Type gamma_et = element::f32;
    element::Type beta_et = element::f32;
    element::Type mean_et = element::f32;
    element::Type variance_et = element::f32;
    element::Type delta_et = element::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);
    auto delta = make_shared<op::Parameter>(delta_et, delta_shape);

    try
    {
        auto bn = make_shared<op::BatchNormTrainingBackprop>(
            data_batch, gamma, beta, mean, variance, delta, epsilon);
        FAIL() << "Inconsistent input/delta spatial dimensions not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("Shape of delta does not match the shape of the input data "
                        "(input data shape: {?,3,?,224}, delta shape: {?,3,?,448})"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
