// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"

using namespace std;
using namespace ov;
using namespace testing;

template <class T>
class ScatterElementsUpdateTest : public ::testing::Test {};
TYPED_TEST_SUITE_P(ScatterElementsUpdateTest);

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_update_output_shape) {
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2, 2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};
    Shape expected_output_shape{2, 4, 5, 7};

    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::v0::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::v0::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::v0::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<TypeParam>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_shape(0), expected_output_shape);
}

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_update_output_partial_dyn_shape) {
    PartialShape data_shape{2, Dimension::dynamic(), 5};
    auto symbols = set_shape_symbols(data_shape);
    PartialShape indices_shape{Dimension::dynamic(), 2, 2};
    PartialShape updates_shape{2, 2, Dimension::dynamic()};
    PartialShape axis_shape = PartialShape::dynamic();

    auto data = make_shared<op::v0::Parameter>(element::f64, data_shape);
    auto indices = make_shared<op::v0::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::v0::Parameter>(element::f64, updates_shape);
    auto axis = make_shared<op::v0::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<TypeParam>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_element_type(0), element::f64);
    EXPECT_EQ(scatter->get_output_partial_shape(0), data_shape);
    EXPECT_THAT(get_shape_symbols(scatter->get_output_partial_shape(0)), symbols);
}

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_update_data_has_interval_dimensions) {
    PartialShape data_shape{{5, 10}, -1, {-1, 3}, {8, -1}};
    auto symbols = set_shape_symbols(data_shape);

    const auto data = make_shared<op::v0::Parameter>(element::i64, data_shape);
    const auto indices = make_shared<op::v0::Parameter>(element::i16, PartialShape{1, 2, 2, {2, 3}});
    const auto updates = make_shared<op::v0::Parameter>(element::i64, PartialShape{{0, 2}, -1, 2, -1});
    const auto axis = make_shared<op::v0::Parameter>(element::i16, PartialShape::dynamic());

    const auto scatter = make_shared<TypeParam>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_element_type(0), element::i64);
    EXPECT_EQ(scatter->get_output_partial_shape(0), data_shape);
    EXPECT_THAT(get_shape_symbols(scatter->get_output_partial_shape(0)), symbols);
}

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_update_output_full_dyn_shape) {
    PartialShape data_shape = PartialShape::dynamic();
    PartialShape indices_shape = PartialShape::dynamic();
    PartialShape updates_shape = PartialShape::dynamic();
    PartialShape axis_shape = PartialShape::dynamic();

    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::v0::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::v0::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::v0::Parameter>(element::i16, axis_shape);

    auto scatter = make_shared<TypeParam>(data, indices, updates, axis);

    EXPECT_EQ(scatter->get_output_element_type(0), element::f32);
    EXPECT_EQ(scatter->get_output_partial_shape(0), data_shape);
}

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_update_default_ctor) {
    const auto data = make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 5, 5, 6});
    const auto indices = make_shared<op::v0::Parameter>(element::i16, PartialShape{1, 2, 1, 3});
    const auto updates = make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 1, 3});
    const auto axis = make_shared<op::v0::Constant>(element::i16, Shape{}, -4);

    const auto scatter = make_shared<TypeParam>(data, indices, updates, axis);
    scatter->set_arguments(OutputVector{data, indices, updates, axis});
    scatter->validate_and_infer_types();

    EXPECT_EQ(scatter->get_input_size(), 4);
    EXPECT_EQ(scatter->get_output_size(), 1);
    EXPECT_EQ(scatter->get_output_element_type(0), element::f32);
    EXPECT_EQ(scatter->get_output_partial_shape(0), PartialShape({2, 5, 5, 6}));
    EXPECT_THAT(get_shape_symbols(scatter->get_output_partial_shape(0)), Each(nullptr));
}

TYPED_TEST_P(ScatterElementsUpdateTest,
             scatter_elements_update_preserve_partial_values_and_labels_via_evaluates_bounds) {
    const auto data = op::v0::Constant::create(element::i64, Shape{4}, {2, 3, 15, 4});
    const auto indices = op::v0::Constant::create(element::i64, Shape{2}, {3, 0});
    auto updates_shape = PartialShape{{10, 20}, {3, 4}};
    auto symbols = set_shape_symbols(updates_shape);
    const auto axis = make_shared<op::v0::Constant>(element::i16, Shape{}, 0);

    const auto shape_of_u =
        std::make_shared<op::v0::ShapeOf>(std::make_shared<op::v0::Parameter>(element::i64, updates_shape));
    const auto scatter = make_shared<TypeParam>(data, indices, shape_of_u, axis);

    auto param = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1});
    auto bc = std::make_shared<op::v3::Broadcast>(param, scatter, op::BroadcastType::BIDIRECTIONAL);

    EXPECT_EQ(bc->get_output_partial_shape(0), PartialShape({{3, 4}, 3, 15, {10, 20}}));
    EXPECT_THAT(get_shape_symbols(bc->get_output_partial_shape(0)),
                ElementsAre(symbols[1], nullptr, nullptr, symbols[0]));
}

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_update_axis_validation) {
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2, 2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::v0::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::v0::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::v0::Constant>(element::i16, axis_shape, std::vector<int>{8});

    OV_EXPECT_THROW(auto scatter = make_shared<TypeParam>(data, indices, updates, axis),
                    ov::AssertFailure,
                    HasSubstr("Axis 8 out of the tensor rank range [-4, 3]"));
}

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_updates_indices_shape) {
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{3, 3, 3, 3};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::v0::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::v0::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::v0::Constant>(element::i16, axis_shape, std::vector<int>{1});

    OV_EXPECT_THROW(auto scatter = make_shared<TypeParam>(data, indices, updates, axis),
                    NodeValidationFailure,
                    HasSubstr("Indices and updates input shapes are required to be equal"));
}

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_updates_indices_rank) {
    Shape data_shape{2, 4};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2, 2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::v0::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::v0::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::v0::Constant>(element::i16, axis_shape, std::vector<int>{1});

    OV_EXPECT_THROW(auto scatter = make_shared<TypeParam>(data, indices, updates, axis),
                    NodeValidationFailure,
                    HasSubstr("Indices and updates input shapes are required to be equal"));
}

TYPED_TEST_P(ScatterElementsUpdateTest, scatter_elements_data_indices_rank) {
    Shape data_shape{2, 4, 5, 7};
    Shape indices_shape{2, 2};
    Shape updates_shape{2, 2};
    Shape axis_shape{};

    auto data = make_shared<op::v0::Parameter>(element::f32, data_shape);
    auto indices = make_shared<op::v0::Parameter>(element::i16, indices_shape);
    auto updates = make_shared<op::v0::Parameter>(element::f32, updates_shape);
    auto axis = make_shared<op::v0::Constant>(element::i16, axis_shape, std::vector<int>{1});

    OV_EXPECT_THROW(auto scatter = make_shared<TypeParam>(data, indices, updates, axis),
                    NodeValidationFailure,
                    HasSubstr("Indices rank and data rank are required to be equal"));
}

TEST(type_prop, scatter_elements_update_mean_reduction_of_bool) {
    const auto data = make_shared<op::v0::Parameter>(element::boolean, Shape{10});
    const auto indices = make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto updates = make_shared<op::v0::Parameter>(element::boolean, Shape{2});
    const auto axis = make_shared<op::v0::Constant>(element::i32, Shape{1}, std::vector<int>{0});

    OV_EXPECT_THROW(
        std::ignore = make_shared<ov::op::v12::ScatterElementsUpdate>(data,
                                                                      indices,
                                                                      updates,
                                                                      axis,
                                                                      op::v12::ScatterElementsUpdate::Reduction::MEAN),
        NodeValidationFailure,
        HasSubstr("The 'mean' reduction type is not supported for boolean tensors"));
}

REGISTER_TYPED_TEST_SUITE_P(ScatterElementsUpdateTest,
                            scatter_elements_update_output_shape,
                            scatter_elements_update_output_partial_dyn_shape,
                            scatter_elements_update_data_has_interval_dimensions,
                            scatter_elements_update_output_full_dyn_shape,
                            scatter_elements_update_default_ctor,
                            scatter_elements_update_preserve_partial_values_and_labels_via_evaluates_bounds,
                            scatter_elements_update_axis_validation,
                            scatter_elements_updates_indices_shape,
                            scatter_elements_updates_indices_rank,
                            scatter_elements_data_indices_rank);

using OpVersions = ::testing::Types<op::v3::ScatterElementsUpdate, op::v12::ScatterElementsUpdate>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, ScatterElementsUpdateTest, OpVersions);
