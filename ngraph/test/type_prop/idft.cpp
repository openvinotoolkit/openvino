// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

struct ConstantAxesAndConstantSignalSizeTestParams
{
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
};

struct ConstantAxesAndConstantSignalSizeTest
    : ::testing::TestWithParam<ConstantAxesAndConstantSignalSizeTestParams>
{
};

TEST_P(ConstantAxesAndConstantSignalSizeTest, idft_constant_axes_and_signal_size)
{
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

    std::shared_ptr<op::v7::IDFT> idft;
    if (params.signal_size.empty())
    {
        idft = std::make_shared<op::v7::IDFT>(data, axes_input);
    }
    else
    {
        auto signal_size_input = op::Constant::create<int64_t>(
            element::i64, params.signal_size_shape, params.signal_size);
        idft = std::make_shared<op::v7::IDFT>(data, axes_input, signal_size_input);
    }

    EXPECT_EQ(idft->get_element_type(), element::f32);
    ASSERT_TRUE(idft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    ConstantAxesAndConstantSignalSizeTest,
    ::testing::Values(
        ConstantAxesAndConstantSignalSizeTestParams{
            {2, 180, 180, 2}, {2}, Shape{}, {2, 180, 180, 2}, {1, 2}, {}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {2, 180, 180, 2}, {2}, Shape{}, {2, 180, 180, 2}, {2, 0}, {}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {16, 500, 180, 369, 2}, {3}, Shape{}, {16, 500, 180, 369, 2}, {0, 3, 1}, {}},
        ConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, Dimension(1, 18)},
                                                    {2},
                                                    Shape{},
                                                    {2, 180, 180, Dimension(1, 18)},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), 2},
                                                    {2},
                                                    Shape{},
                                                    {2, 180, Dimension(7, 500), 2},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
                                                    {2},
                                                    Shape{},
                                                    {2, 180, Dimension(7, 500), Dimension(1, 18)},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, 2},
                                                    {2},
                                                    Shape{},
                                                    {2, Dimension(7, 500), 180, 2},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
                                                    {2},
                                                    Shape{},
                                                    {2, Dimension(7, 500), 180, Dimension(1, 18)},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
                                                    {2},
                                                    Shape{},
                                                    {2, Dimension(7, 500), Dimension(7, 500), 2},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            Shape{},
            {2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {1, 2},
            {}},
        ConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, 2},
                                                    {2},
                                                    Shape{},
                                                    {Dimension(0, 2), 180, 180, 2},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                                    {2},
                                                    Shape{},
                                                    {Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
                                                    {2},
                                                    Shape{},
                                                    {Dimension(0, 2), 180, Dimension(7, 500), 2},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
            {2},
            Shape{},
            {Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
            {1, 2},
            {}},
        ConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
                                                    {2},
                                                    Shape{},
                                                    {Dimension(0, 2), Dimension(7, 500), 180, 2},
                                                    {1, 2},
                                                    {}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
            {2},
            Shape{},
            {Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
            {1, 2},
            {}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
            {2},
            Shape{},
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
            {1, 2},
            {}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            Shape{},
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {1, 2},
            {}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {2, 180, 180, 2}, {2}, {2}, {2, 180, 77, 2}, {1, 2}, {-1, 77}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {2, 180, 180, 2}, {2}, {2}, {87, 180, 390, 2}, {2, 0}, {390, 87}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {7, 50, 130, 400, 2}, {3}, {3}, {7, 40, 130, 600, 2}, {3, 0, 1}, {600, -1, 40}},
        ConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                                    {2},
                                                    {2},
                                                    {2, Dimension(0, 200), 77, 2},
                                                    {1, 2},
                                                    {-1, 77}},
        ConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                                    {2},
                                                    {2},
                                                    {87, 180, 390, 2},
                                                    {2, 0},
                                                    {390, 87}},
        ConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
            {3},
            {3},
            {Dimension(8, 129), 40, 130, 600, 2},
            {3, 0, 1},
            {600, -1, 40}}),
    PrintToDummyParamName());

TEST(type_prop, idft_dynamic_axes)
{
    const auto input_shape = PartialShape{2, 180, 180, Dimension(1, 18)};
    const auto axes_shape = PartialShape::dynamic();
    const auto ref_output_shape = PartialShape{
        Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)};

    auto data = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto axes_input = std::make_shared<op::Parameter>(element::i64, axes_shape);
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    EXPECT_EQ(idft->get_element_type(), element::f32);
    ASSERT_TRUE(idft->get_output_partial_shape(0).same_scheme(ref_output_shape));
}

struct NonConstantAxesTestParams
{
    PartialShape input_shape;
    Shape axes_shape;
    PartialShape ref_output_shape;
};

struct NonConstantAxesTest : ::testing::TestWithParam<NonConstantAxesTestParams>
{
};

TEST_P(NonConstantAxesTest, idft_non_constant_axes)
{
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = std::make_shared<op::Parameter>(element::i64, params.axes_shape);
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input);

    EXPECT_EQ(idft->get_element_type(), element::f32);
    ASSERT_TRUE(idft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    NonConstantAxesTest,
    ::testing::Values(
        NonConstantAxesTestParams{
            {2, 180, 180, Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{
            {2, 180, Dimension(7, 500), 2},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{
            {2, 180, Dimension(7, 500), Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{
            {2, Dimension(7, 500), 180, 2},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{
            {2, Dimension(7, 500), 180, Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{
            {2, Dimension(7, 500), Dimension(7, 500), 2},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{
            {2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), 180, 180, 2},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), 180, 180, Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), 180, Dimension(7, 500), 2},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), Dimension(7, 500), 180, 2},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}}),
    PrintToDummyParamName());

struct NonConstantSignalSizeTestParams
{
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
};

struct NonConstantSignalSizeTest : ::testing::TestWithParam<NonConstantSignalSizeTestParams>
{
};

TEST_P(NonConstantSignalSizeTest, idft_non_constant_signal_size)
{
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
    auto signal_size_input =
        std::make_shared<op::Parameter>(element::i64, params.signal_size_shape);
    auto idft = std::make_shared<op::v7::IDFT>(data, axes_input, signal_size_input);

    EXPECT_EQ(idft->get_element_type(), element::f32);
    ASSERT_TRUE(idft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    NonConstantSignalSizeTest,
    ::testing::Values(
        NonConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                        {2},
                                        {2},
                                        {2, Dimension::dynamic(), Dimension::dynamic(), 2},
                                        {1, 2}},
        NonConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                        {2},
                                        {2},
                                        {Dimension::dynamic(), 180, Dimension::dynamic(), 2},
                                        {2, 0}},
        NonConstantSignalSizeTestParams{
            {Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
            {3},
            {3},
            {Dimension::dynamic(), Dimension::dynamic(), 130, Dimension::dynamic(), 2},
            {3, 0, 1}}),
    PrintToDummyParamName());
