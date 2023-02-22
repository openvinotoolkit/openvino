// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;
using namespace testing;

TEST(type_prop, scatter_elements_update_output_shape) {
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

TEST(type_prop, scatter_elements_update_output_partial_dyn_shape) {
    PartialShape data_shape{2, Dimension::dynamic(), 5};
    set_shape_labels(data_shape, 10);
    PartialShape indices_shape{Dimension::dynamic(), 2, 2};
    PartialShape updates_shape{2, 2, Dimension::dynamic()};
    PartialShape axis_shape = PartialShape::dynamic();

    auto data = make_shared<op::Parameter>(element::f64, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f64, updates_shape);
    auto axis = make_shared<op::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_element_type(0), element::f64);
    EXPECT_EQ(scatter->get_output_partial_shape(0), data_shape);
    EXPECT_THAT(get_shape_labels(scatter->get_output_partial_shape(0)), ElementsAre(10, 11, 12));
}

TEST(type_prop, scatter_elements_update_data_has_interval_dimensions) {
    PartialShape data_shape{{5, 10}, -1, {-1, 3}, {8, -1}};
    set_shape_labels(data_shape, 10);
    PartialShape indices_shape{1, 2, 2, {2, 3}};
    PartialShape updates_shape{{0, 2}, -1, 2, -1};
    PartialShape axis_shape = PartialShape::dynamic();

    auto data = make_shared<op::Parameter>(element::i64, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::i64, updates_shape);
    auto axis = make_shared<op::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_element_type(0), element::i64);
    EXPECT_EQ(scatter->get_output_partial_shape(0), data_shape);
    EXPECT_THAT(get_shape_labels(scatter->get_output_partial_shape(0)), ElementsAre(10, 11, 12, 13));
}

TEST(type_prop, scatter_elements_update_output_full_dyn_shape) {
    PartialShape data_shape = PartialShape::dynamic();
    PartialShape indices_shape = PartialShape::dynamic();
    PartialShape updates_shape = PartialShape::dynamic();
    PartialShape axis_shape = PartialShape::dynamic();

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_element_type(0), element::f32);
    EXPECT_EQ(scatter->get_output_partial_shape(0), data_shape);
}

TEST(type_prop, scatter_elements_update_axis_validation) {
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2, 2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, std::vector<int>{8});

    try {
        auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis);
        FAIL() << "Not detected axis with value out of the range";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Axis value has to be in range"));
    } catch (...) {
        FAIL() << "Deduced type check failed for unexpected reason";
    }
}

TEST(type_prop, scatter_elements_updates_indices_shape) {
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{3, 3, 3, 3};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, std::vector<int>{1});

    OV_EXPECT_THROW(auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis),
                    NodeValidationFailure,
                    HasSubstr("Indices and updates input shapes are required to be equal"));
}

TEST(type_prop, scatter_elements_updates_indices_rank) {
    Shape data_shape{2, 4};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, std::vector<int>{1});

    OV_EXPECT_THROW(auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis),
                    NodeValidationFailure,
                    HasSubstr("Indices and updates input shapes are required to be equal"));
}

TEST(type_prop, scatter_elements_data_indices_rank) {
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::Constant>(element::i16, axis_shape, std::vector<int>{1});

    OV_EXPECT_THROW(auto scatter = make_shared<op::v3::ScatterElementsUpdate>(data, indices, updates, axis),
                    NodeValidationFailure,
                    HasSubstr("Indices rank and data rank are required to be equal"));
}
