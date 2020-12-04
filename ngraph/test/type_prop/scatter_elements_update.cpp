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

TEST(type_prop, scatter_elements_update_output_shape)
{
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2, 2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};
    Shape expected_output_shape{2, 4, 5, 7};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_shape(0), expected_output_shape);
}

TEST(type_prop, scatter_elements_update_output_partial_dyn_shape)
{
    PartialShape data_shape{2, Dimension::dynamic(), 5};
    PartialShape indices_shape{Dimension::dynamic(), 2, 2};
    PartialShape updates_shape{2, 2, Dimension::dynamic()};
    PartialShape axis_shape = PartialShape::dynamic();

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);

    EXPECT_TRUE(scatter->get_output_partial_shape(0).same_scheme(data_shape));
}

TEST(type_prop, scatter_elements_update_output_full_dyn_shape)
{
    PartialShape data_shape = PartialShape::dynamic();
    PartialShape indices_shape = PartialShape::dynamic();
    PartialShape updates_shape = PartialShape::dynamic();
    PartialShape axis_shape = PartialShape::dynamic();

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);

    EXPECT_TRUE(scatter->get_output_partial_shape(0).same_scheme(data_shape));
}

TEST(type_prop, scatter_elements_update_axis_validation)
{
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2, 2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, std::vector<int>{8});

    try
    {
        auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
        FAIL() << "Not detected axis with value out of the range";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Axis value has to be in range"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_elements_updates_indices_shape)
{
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{3, 3, 3, 3};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, std::vector<int>{1});

    try
    {
        auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
        FAIL() << "Not detected incompatibile indices and updates shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Indices and updates input shapes are required to be equal"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_elements_updates_indices_rank)
{
    Shape data_shape{2, 4};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, std::vector<int>{1});

    try
    {
        auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
        FAIL() << "Not detected incompatibile indices and updates shape";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(
            error.what(), std::string("Indices and updates input shapes are required to be equal"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_elements_data_indices_rank)
{
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, std::vector<int>{1});

    try
    {
        auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
        FAIL() << "Not detected incompatibile indices and data rank";
    }
    catch (const NodeValidationFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("Indices rank and data rank are required to be equal"));
    }
    catch (...)
    {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}
