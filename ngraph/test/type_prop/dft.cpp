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

TEST_P(ConstantAxesAndConstantSignalSizeTest, dft_constant_axes_and_signal_size)
{
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

    std::shared_ptr<op::v7::DFT> dft;
    if (params.signal_size.empty())
    {
        dft = std::make_shared<op::v7::DFT>(data, axes_input);
    }
    else
    {
        auto signal_size_input = op::Constant::create<int64_t>(
            element::i64, params.signal_size_shape, params.signal_size);
        dft = std::make_shared<op::v7::DFT>(data, axes_input, signal_size_input);
    }

    EXPECT_EQ(dft->get_element_type(), element::f32);
    ASSERT_TRUE(dft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_CASE_P(
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
            {}}),
    PrintToDummyParamName());


// TEST_P(ConstantAxesAndConstantSignalSizeTest, dft_constant_axes_and_constant_signal_size)
// {
//     auto params = GetParam();
//
//     auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
//     auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
//     auto signal_size_input =
//         op::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
//     auto dft = std::make_shared<op::v7::DFT>(data, axes_input, signal_size_input);
//
//     EXPECT_EQ(dft->get_element_type(), element::f32);
//     ASSERT_TRUE(dft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
// }
//

// struct ConstantAxesAndNoSignalSizeTestParams
// {
//     PartialShape input_shape;
//     Shape axes_shape;
//     PartialShape ref_output_shape;
//     std::vector<int64_t> axes;
// };
//
// struct ConstantAxesAndNoSignalSizeTest
//     : ::testing::TestWithParam<ConstantAxesAndNoSignalSizeTestParams>
// {
// };
//
// TEST_P(ConstantAxesAndNoSignalSizeTest, dft_constant_axes_there_are_no_signal_size)
// {
//     auto params = GetParam();
//
//     auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
//     auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
//     auto dft = std::make_shared<op::v7::DFT>(data, axes_input);
//
//     EXPECT_EQ(dft->get_element_type(), element::f32);
//     ASSERT_TRUE(dft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
// }
//
// INSTANTIATE_TEST_CASE_P(
//     type_prop,
//     ConstantAxesAndNoSignalSizeTest,
//     ::testing::Values(
//         ConstantAxesAndNoSignalSizeTestParams{{2, 180, 180, 2}, {2}, {2, 180, 180, 2}, {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{{2, 180, 180, 2}, {2}, {2, 180, 180, 2}, {2, 0}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {16, 500, 180, 369, 2}, {3}, {16, 500, 180, 369, 2}, {0, 3, 1}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {2, 180, 180, Dimension(1, 18)}, {2}, {2, 180, 180, Dimension(1, 18)}, {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {2, 180, Dimension(7, 500), 2}, {2}, {2, 180, Dimension(7, 500), 2}, {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
//                                               {2},
//                                               {2, 180, Dimension(7, 500), Dimension(1, 18)},
//                                               {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {2, Dimension(7, 500), 180, 2}, {2}, {2, Dimension(7, 500), 180, 2}, {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
//                                               {2},
//                                               {2, Dimension(7, 500), 180, Dimension(1, 18)},
//                                               {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
//                                               {2},
//                                               {2, Dimension(7, 500), Dimension(7, 500), 2},
//                                               {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
//             {2},
//             {2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
//             {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {Dimension(0, 2), 180, 180, 2}, {2}, {Dimension(0, 2), 180, 180, 2}, {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
//                                               {2},
//                                               {Dimension(0, 2), 180, 180, Dimension(1, 18)},
//                                               {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
//                                               {2},
//                                               {Dimension(0, 2), 180, Dimension(7, 500), 2},
//                                               {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
//             {2},
//             {Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
//             {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
//                                               {2},
//                                               {Dimension(0, 2), Dimension(7, 500), 180, 2},
//                                               {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
//             {2},
//             {Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
//             {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
//             {2},
//             {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
//             {1, 2}},
//         ConstantAxesAndNoSignalSizeTestParams{
//             {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
//             {2},
//             {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
//             {1, 2}}),
//     PrintToDummyParamName());
//
// struct ConstantAxesAndConstantSignalSizeTestParams
// {
//     PartialShape input_shape;
//     Shape axes_shape;
//     Shape signal_size_shape;
//     PartialShape ref_output_shape;
//     std::vector<int64_t> axes;
//     std::vector<int64_t> signal_size;
// };
//
// struct ConstantAxesAndConstantSignalSizeTest
//     : ::testing::TestWithParam<ConstantAxesAndConstantSignalSizeTestParams>
// {
// };
//
// TEST_P(ConstantAxesAndConstantSignalSizeTest, dft_constant_axes_and_constant_signal_size)
// {
//     auto params = GetParam();
//
//     auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
//     auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
//     auto signal_size_input =
//         op::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
//     auto dft = std::make_shared<op::v7::DFT>(data, axes_input, signal_size_input);
//
//     EXPECT_EQ(dft->get_element_type(), element::f32);
//     ASSERT_TRUE(dft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
// }
//
// INSTANTIATE_TEST_CASE_P(
//     type_prop,
//     ConstantAxesAndConstantSignalSizeTest,
//     ::testing::Values(
//         ConstantAxesAndConstantSignalSizeTestParams{
//             {2, 180, 180, 2}, {2}, {2}, {2, 180, 77, 2}, {1, 2}, {-1, 77}},
//         ConstantAxesAndConstantSignalSizeTestParams{
//             {2, 180, 180, 2}, {2}, {2}, {87, 180, 390, 2}, {2, 0}, {390, 87}},
//         ConstantAxesAndConstantSignalSizeTestParams{
//             {7, 50, 130, 400, 2}, {3}, {3}, {7, 40, 130, 600, 2}, {3, 0, 1}, {600, -1, 40}},
//         ConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
//                                                     {2},
//                                                     {2},
//                                                     {2, Dimension(0, 200), 77, 2},
//                                                     {1, 2},
//                                                     {-1, 77}},
//         ConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
//                                                     {2},
//                                                     {2},
//                                                     {87, 180, 390, 2},
//                                                     {2, 0},
//                                                     {390, 87}},
//         ConstantAxesAndConstantSignalSizeTestParams{
//             {Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
//             {3},
//             {3},
//             {Dimension(8, 129), 40, 130, 600, 2},
//             {3, 0, 1},
//             {600, -1, 40}}),
//     PrintToDummyParamName());

TEST(type_prop, dft_constant_axes_and_there_are_no_signal_size_dynamic_shapes2)
{
    struct ShapesAndValues
    {
        PartialShape input_shape;
        Shape axes_shape;
        PartialShape ref_output_shape;
    };

    std::vector<ShapesAndValues> shapes_and_values = {
        {{2, 180, 180, Dimension(1, 18)}, {2}, {2, 180, 180, Dimension(1, 18)}},
        {{2, 180, Dimension(7, 500), 2}, {2}, {2, 180, Dimension(7, 500), 2}},
        {{2, 180, Dimension(7, 500), Dimension(1, 18)},
         {2},
         {2, 180, Dimension(7, 500), Dimension(1, 18)}},
        {{2, Dimension(7, 500), 180, 2}, {2}, {2, Dimension(7, 500), 180, 2}},
        {{2, Dimension(7, 500), 180, Dimension(1, 18)},
         {2},
         {2, Dimension(7, 500), 180, Dimension(1, 18)}},
        {{2, Dimension(7, 500), Dimension(7, 500), 2},
         {2},
         {2, Dimension(7, 500), Dimension(7, 500), 2}},
        {{2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
         {2},
         {2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)}},
        {{Dimension(0, 2), 180, 180, 2}, {2}, {Dimension(0, 2), 180, 180, 2}},
        {{Dimension(0, 2), 180, 180, Dimension(1, 18)},
         {2},
         {Dimension(0, 2), 180, 180, Dimension(1, 18)}},
        {{Dimension(0, 2), 180, Dimension(7, 500), 2},
         {2},
         {Dimension(0, 2), 180, Dimension(7, 500), 2}},
        {{Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
         {2},
         {Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)}},
        {{Dimension(0, 2), Dimension(7, 500), 180, 2},
         {2},
         {Dimension(0, 2), Dimension(7, 500), 180, 2}},
        {{Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
         {2},
         {Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)}},
        {{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
         {2},
         {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2}},
        {{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
         {2},
         {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)}}};

    for (const auto& s : shapes_and_values)
    {
        auto data = std::make_shared<op::Parameter>(element::f32, s.input_shape);
        auto axes_input = std::make_shared<op::Parameter>(element::i64, s.axes_shape);
        auto dft = std::make_shared<op::v7::DFT>(data, axes_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        ASSERT_TRUE(dft->get_output_partial_shape(0).same_scheme(s.ref_output_shape));
    }
}

TEST(type_prop, dft_constant_axes_and_there_are_signal_size_dynamic_shapes2)
{
    struct ShapesAndValues
    {
        PartialShape input_shape;
        Shape axes_shape;
        Shape signal_size_shape;
        PartialShape ref_output_shape;
        std::vector<int64_t> axes;
    };

    std::vector<ShapesAndValues> shapes_and_values = {
        {{2, Dimension(0, 200), 180, 2}, {2}, {2}, {2, Dimension(0, 200), 180, 2}, {1, 2}},
        {{Dimension(0, 18), 180, Dimension(0, 400), 2},
         {2},
         {2},
         {Dimension(0, 18), 180, Dimension(0, 400), 2},
         {2, 0}},
        {{Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
         {3},
         {3},
         {Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
         {3, 0, 1}}};

    for (const auto& s : shapes_and_values)
    {
        auto data = std::make_shared<op::Parameter>(element::f32, s.input_shape);
        auto axes_input = op::Constant::create<int64_t>(element::i64, s.axes_shape, s.axes);
        auto signal_size_input = std::make_shared<op::Parameter>(element::i64, s.signal_size_shape);
        auto dft = std::make_shared<op::v7::DFT>(data, axes_input, signal_size_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        ASSERT_TRUE(dft->get_output_partial_shape(0).same_scheme(s.ref_output_shape));
    }
}
