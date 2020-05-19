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

TEST(type_prop, dyn_broadcast_shape_wrong_rank)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = make_shared<op::Parameter>(element::i64, Shape{1, 1});
    auto bc_axes = make_shared<op::Parameter>(element::i64, Shape{1});

    try
    {
        auto bc = make_shared<op::DynBroadcast>(arg, bc_shape, bc_axes);
        FAIL() << "DynBroadcast: wrong shape rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "DynBroadcast shape rank must be 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dyn_broadcast_axes_wrong_rank)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = make_shared<op::Parameter>(element::i64, Shape{1});
    auto bc_axes = make_shared<op::Parameter>(element::i64, Shape{2, 2});

    try
    {
        auto bc = make_shared<op::DynBroadcast>(arg, bc_shape, bc_axes);
        FAIL() << "DynBroadcast: axes shape rank not detected";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), "DynBroadcast axes rank must be 1");
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dyn_broadcast_output_partial_shape_dynamic)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = make_shared<op::Parameter>(element::i64, Shape{1});
    auto bc_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    auto bc = make_shared<op::DynBroadcast>(arg, bc_shape, bc_axes);
    ASSERT_TRUE(bc->get_output_partial_shape(0).is_dynamic());
}

TEST(type_prop, dyn_broadcast_broadcast_shape_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    // wrong element type
    auto bc_shape = make_shared<op::Parameter>(element::boolean, Shape{1});
    auto bc_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    try
    {
        auto bc = make_shared<op::DynBroadcast>(arg, bc_shape, bc_axes);
        FAIL() << "DynBroadcast: did not detect shape element type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("DynBroadcast shape must have element type i64"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, dyn_broadcast_axes_et_wrong)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto bc_shape = make_shared<op::Parameter>(element::i64, Shape{1});
    // wrong element type
    auto bc_axes = make_shared<op::Parameter>(element::f32, Shape{2});

    try
    {
        auto bc = make_shared<op::DynBroadcast>(arg, bc_shape, bc_axes);
        FAIL() << "DynBroadcast: did not detect axes element type not i64";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("DynBroadcast axes must have element type i64"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
