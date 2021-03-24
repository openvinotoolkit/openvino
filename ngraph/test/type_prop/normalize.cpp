// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, normalize_axes_input_not_constant)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto axes = make_shared<op::Parameter>(element::u64, Shape{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    try
    {
        auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Input axes must be Constant type"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize_invalid_axes_rank)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{1, 2}, vector<int64_t>{1, 2});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    try
    {
        auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Input axes must be scalar or have rank equal to 1"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, normalize_axes_out_of_bounds)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{3, 4});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    try
    {
        auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
        // Should have thrown, so fail if it didn't
        FAIL() << "Invalid input tensor rank.";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Reduction axis ("));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
