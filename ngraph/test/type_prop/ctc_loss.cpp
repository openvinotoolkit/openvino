//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

TEST(type_prop, ctc_loss)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss =
        make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_TRUE(ctc_loss->get_output_partial_shape(0).same_scheme(PartialShape{10}));
}

TEST(type_prop, ctc_loss_no_blank_index)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});

    // create CTCLoss node
    auto ctc_loss = make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_TRUE(ctc_loss->get_output_partial_shape(0).same_scheme(PartialShape{10}));
}

TEST(type_prop, ctc_loss_output_type)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f64, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss =
        make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f64);
    EXPECT_TRUE(ctc_loss->get_output_partial_shape(0).same_scheme(PartialShape{10}));
}

TEST(type_prop, ctc_loss_non_default_parameters)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f64, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss = make_shared<op::v4::CTCLoss>(
        logits, logit_length, labels, label_length, blank_index, true, false, false);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f64);
    EXPECT_TRUE(ctc_loss->get_output_partial_shape(0).same_scheme(PartialShape{10}));
}

TEST(type_prop, ctc_loss_dynamic_input)
{
    // create inputs
    auto logits =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 120, 28});
    auto logit_length =
        make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto labels = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic(), 120});
    auto label_length =
        make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss =
        make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_TRUE(
        ctc_loss->get_output_partial_shape(0).same_scheme(PartialShape{Dimension::dynamic()}));
}

TEST(type_prop, ctc_loss_partly_dynamic_input)
{
    // create inputs
    auto logits =
        make_shared<op::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, PartialShape{10});
    auto labels = make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic(), 120});
    auto label_length =
        make_shared<op::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    // create CTCLoss node
    auto ctc_loss =
        make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

    // check type and shape infer
    EXPECT_EQ(ctc_loss->get_element_type(), element::f32);
    EXPECT_TRUE(ctc_loss->get_output_partial_shape(0).same_scheme(PartialShape{10}));
}

TEST(type_prop, ctc_loss_fail_inputs_dim)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 40, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    try
    {
        // create CTCLoss node
        auto ctc_loss =
            make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid inputs not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Expected a 3D tensor for logits."));
    }
    catch (...)
    {
        FAIL() << "Inputs shape check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_loss_fail_logit_length_dim)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10, 20});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    try
    {
        // create CTCLoss node
        auto ctc_loss =
            make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid logit length not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Expected a 1D tensor for logit length."));
    }
    catch (...)
    {
        FAIL() << "Logit length shape check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_loss_fail_labels_dim)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    try
    {
        // create CTCLoss node
        auto ctc_loss =
            make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid labels not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Expected a 2D tensor for labels."));
    }
    catch (...)
    {
        FAIL() << "Labels shape check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_loss_fail_label_length_dim)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10, 40});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    try
    {
        // create CTCLoss node
        auto ctc_loss =
            make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid labels not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Expected a 1D tensor for label length."));
    }
    catch (...)
    {
        FAIL() << "Label length shape check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_loss_fail_blank_index_dim)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{4});

    try
    {
        // create CTCLoss node
        auto ctc_loss =
            make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid labels not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Expected a scalar for blank index."));
    }
    catch (...)
    {
        FAIL() << "Blank index shape check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_loss_fail_batch_dim_mismatch)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 120});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{40});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    try
    {
        // create CTCLoss node
        auto ctc_loss =
            make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of batch dimension not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("The first dimension of label length must be equal to the first dimension "
                        "of the logits, the logit length and labels."));
    }
    catch (...)
    {
        FAIL() << "Batch dimension matching check failed for unexpected reason";
    }
}

TEST(type_prop, ctc_loss_fail_time_dim_mismatch)
{
    // create inputs
    auto logits = make_shared<op::Parameter>(element::f32, Shape{10, 120, 28});
    auto logit_length = make_shared<op::Parameter>(element::i32, Shape{10});
    auto labels = make_shared<op::Parameter>(element::i32, Shape{10, 130});
    auto label_length = make_shared<op::Parameter>(element::i32, Shape{40});
    auto blank_index = make_shared<op::Parameter>(element::i32, Shape{});

    try
    {
        // create CTCLoss node
        auto ctc_loss =
            make_shared<op::v4::CTCLoss>(logits, logit_length, labels, label_length, blank_index);

        // Should have thrown, so fail if it didn't
        FAIL() << "Mismatch of time dimension not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(),
            std::string("The second dimension of labels must be equal to the second dimension "
                        "of logits."));
    }
    catch (...)
    {
        FAIL() << "Time dimension matching check failed for unexpected reason";
    }
}
