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

TEST(type_prop, batch_norm_inference_partial_all_rank_dynamic)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn =
        make_shared<op::v0::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

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
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn =
        make_shared<op::v0::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

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
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v0::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn =
        make_shared<op::v0::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

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
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v0::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v0::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v0::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn =
        make_shared<op::v0::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

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
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v0::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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

TEST(type_prop, batch_norm_inference_partial_all_rank_dynamic_v5)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn =
        make_shared<op::v5::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, batch_norm_inference_partial_input_rank_static_dynamic_ok_v5)
{
    PartialShape data_batch_shape{
        64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn =
        make_shared<op::v5::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}));
}

TEST(type_prop, batch_norm_inference_partial_input_rank_static_dynamic_zero_channels_v5)
{
    PartialShape data_batch_shape{
        Dimension::dynamic(), 0, Dimension::dynamic(), Dimension::dynamic()};
    PartialShape gamma_shape{PartialShape::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{PartialShape::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v5::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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

TEST(type_prop, batch_norm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_ok_v5)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn =
        make_shared<op::v5::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop,
     batch_norm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_wrong_rank_v5)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic(), Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v5::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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
     batch_norm_inference_partial_input_rank_dynamic_some_rank_static_dynamic_inconsistent_rank_v5)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3, Dimension::dynamic()};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{Dimension::dynamic()};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v5::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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
     batch_norm_inference_partial_input_rank_dynamic_some_static_inconsistent_channel_count_v5)
{
    PartialShape data_batch_shape{PartialShape::dynamic()};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{4};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v5::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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

TEST(type_prop, batch_norm_inference_partial_input_rank_static_dynamic_some_static_ok_v5)
{
    PartialShape data_batch_shape{64, Dimension::dynamic(), Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    auto bn =
        make_shared<op::v5::BatchNormInference>(data_batch, gamma, beta, mean, variance, epsilon);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), data_batch_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        PartialShape{64, 3, Dimension::dynamic(), 224}));
}

TEST(
    type_prop,
    batch_norm_inference_partial_input_rank_static_dynamic_some_static_inconsistent_channel_count_v5)
{
    PartialShape data_batch_shape{64, 4, Dimension::dynamic(), 224};
    PartialShape gamma_shape{3};
    PartialShape beta_shape{PartialShape::dynamic()};
    PartialShape mean_shape{3};
    PartialShape variance_shape{PartialShape::dynamic()};
    double epsilon = 0.001;
    element::Type data_batch_et = element::Type_t::f32;
    element::Type gamma_et = element::Type_t::f32;
    element::Type beta_et = element::Type_t::f32;
    element::Type mean_et = element::Type_t::f32;
    element::Type variance_et = element::Type_t::f32;

    auto data_batch = make_shared<op::Parameter>(data_batch_et, data_batch_shape);
    auto gamma = make_shared<op::Parameter>(gamma_et, gamma_shape);
    auto beta = make_shared<op::Parameter>(beta_et, beta_shape);
    auto mean = make_shared<op::Parameter>(mean_et, mean_shape);
    auto variance = make_shared<op::Parameter>(variance_et, variance_shape);

    try
    {
        auto bn = make_shared<op::v5::BatchNormInference>(
            data_batch, gamma, beta, mean, variance, epsilon);
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
