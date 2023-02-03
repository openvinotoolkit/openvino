//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
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

struct IRDFTConstantAxesAndConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
};

struct IRDFTConstantAxesAndConstantSignalSizeTest
    : ::testing::TestWithParam<IRDFTConstantAxesAndConstantSignalSizeTestParams> {};

TEST_P(IRDFTConstantAxesAndConstantSignalSizeTest, irdft_constant_axes_and_signal_size) {
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

    std::shared_ptr<op::v9::IRDFT> irdft;
    if (params.signal_size.empty()) {
        irdft = std::make_shared<op::v9::IRDFT>(data, axes_input);
    } else {
        auto signal_size_input =
            op::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
        irdft = std::make_shared<op::v9::IRDFT>(data, axes_input, signal_size_input);
    }

    EXPECT_EQ(irdft->get_element_type(), element::f32);
    ASSERT_TRUE(irdft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    IRDFTConstantAxesAndConstantSignalSizeTest,
    ::testing::Values(
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, Shape{}, {2, 180, 358}, {1, 2}, {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, Shape{}, {2, 180, 180}, {2, 0}, {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{16, 500, 180, 369, 2},
                                                         {3},
                                                         Shape{},
                                                         {16, 998, 180, 369},
                                                         {0, 3, 1},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {2, 180, 358},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), 2},
                                                         {2},
                                                         Shape{},
                                                         {2, 180, Dimension(12, 998)},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {2, 180, Dimension(12, 998)},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {2, Dimension(7, 500), 358},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {2, Dimension(7, 500), 358},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
                                                         {2},
                                                         Shape{},
                                                         {2, Dimension(7, 500), Dimension(12, 998)},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {2, Dimension(7, 500), Dimension(12, 998)},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), 180, 358},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), 180, 358},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), 180, Dimension(12, 998)},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), 180, Dimension(12, 998)},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), Dimension(7, 500), 358},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), Dimension(7, 500), 358},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                                         {2},
                                                         Shape{},
                                                         {Dimension(0, 2), Dimension(7, 500), Dimension(12, 998)},
                                                         {1, 2},
                                                         {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            Shape{},
            {Dimension(0, 2), Dimension(7, 500), Dimension(12, 998)},
            {1, 2},
            {}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, {2}, {2, 180, 77}, {1, 2}, {-1, 77}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, {2}, {87, 180, 390}, {2, 0}, {390, 87}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{7, 50, 130, 400, 2},
                                                         {3},
                                                         {3},
                                                         {7, 40, 130, 600},
                                                         {3, 0, 1},
                                                         {600, -1, 40}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                                         {2},
                                                         {2},
                                                         {2, Dimension(0, 200), 77},
                                                         {1, 2},
                                                         {-1, 77}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                                         {2},
                                                         {2},
                                                         {87, 180, 390},
                                                         {2, 0},
                                                         {390, 87}},
        IRDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                                                         {3},
                                                         {3},
                                                         {Dimension(8, 129), 40, 130, 600},
                                                         {3, 0, 1},
                                                         {600, -1, 40}}),
    PrintToDummyParamName());

TEST(type_prop, irdft_dynamic_axes) {
    const auto input_shape = PartialShape{2, 180, 180, Dimension(1, 18)};
    const auto axes_shape = PartialShape::dynamic();
    const auto ref_output_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};

    auto data = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto axes_input = std::make_shared<op::Parameter>(element::i64, axes_shape);
    auto irdft = std::make_shared<op::v9::IRDFT>(data, axes_input);

    EXPECT_EQ(irdft->get_element_type(), element::f32);
    ASSERT_TRUE(irdft->get_output_partial_shape(0).same_scheme(ref_output_shape));
}

struct IRDFTNonConstantAxesTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    PartialShape ref_output_shape;
};

struct IRDFTNonConstantAxesTest : ::testing::TestWithParam<IRDFTNonConstantAxesTestParams> {};

TEST_P(IRDFTNonConstantAxesTest, irdft_non_constant_axes) {
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = std::make_shared<op::Parameter>(element::i64, params.axes_shape);
    auto irdft = std::make_shared<op::v9::IRDFT>(data, axes_input);

    EXPECT_EQ(irdft->get_element_type(), element::f32);
    ASSERT_TRUE(irdft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    IRDFTNonConstantAxesTest,
    ::testing::Values(
        IRDFTNonConstantAxesTestParams{{2, 180, 180, Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, 180, Dimension(7, 500), 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, Dimension(7, 500), 180, 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, 180, 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}},
        IRDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                       {2},
                                       {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()}}),
    PrintToDummyParamName());

struct IRDFTNonConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
};

struct IRDFTNonConstantSignalSizeTest : ::testing::TestWithParam<IRDFTNonConstantSignalSizeTestParams> {};

TEST_P(IRDFTNonConstantSignalSizeTest, irdft_non_constant_signal_size) {
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
    auto signal_size_input = std::make_shared<op::Parameter>(element::i64, params.signal_size_shape);
    auto irdft = std::make_shared<op::v9::IRDFT>(data, axes_input, signal_size_input);

    EXPECT_EQ(irdft->get_element_type(), element::f32);
    ASSERT_TRUE(irdft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    IRDFTNonConstantSignalSizeTest,
    ::testing::Values(IRDFTNonConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                                           {2},
                                                           {2},
                                                           {2, Dimension(0, 200), Dimension::dynamic()},
                                                           {1, 2}},
                      IRDFTNonConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                                           {2},
                                                           {2},
                                                           {Dimension::dynamic(), 180, Dimension(0, 400)},
                                                           {2, 0}},
                      IRDFTNonConstantSignalSizeTestParams{
                          {Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                          {3},
                          {3},
                          {Dimension(8, 129), Dimension::dynamic(), 130, Dimension(0, 500)},
                          {3, 0, 1}}),
    PrintToDummyParamName());

TEST(type_prop, irdft_invalid_input) {
    auto axes = op::Constant::create(element::i64, Shape{2}, {0, 1});

    try {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{2});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes);
        FAIL() << "IRDFT node was created with invalid input.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The input rank must be greater or equal to 2.");
    }

    try {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{4, 3});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes);
        FAIL() << "IRDFT node was created with invalid input.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The last dimension of input data must be 2.");
    }

    try {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{4, 2});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes);
        FAIL() << "IRDFT node was created with invalid input.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The input rank must be greater than number of IRDFT op axes.");
    }
}

TEST(type_prop, irdft_invalid_axes) {
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});

    try {
        auto axes = op::Constant::create(element::i64, Shape{1}, {3});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes);
        FAIL() << "IRDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axis 3 must be in the input rank range");
    }

    try {
        auto axes = op::Constant::create(element::i64, Shape{1}, {-3});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes);
        FAIL() << "IRDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axis -3 must be in the input rank range");
    }

    try {
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, -2});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes);
        FAIL() << "IRDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axes must be unique.");
    }

    try {
        auto axes = op::Constant::create(element::i64, Shape{1}, {2});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes);
        FAIL() << "IRDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axis 2 must be in the input rank range");
    }

    try {
        auto axes = op::Constant::create(element::i64, Shape{1, 2}, {0, 1});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes);
        FAIL() << "IRDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axes input must be 1D tensor.");
    }
}

TEST(type_prop, irdft_invalid_signal_size) {
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto axes = op::Constant::create(element::i64, Shape{1}, {0});

    try {
        auto signal_size = op::Constant::create(element::i64, Shape{1, 2}, {0, 1});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes, signal_size);
        FAIL() << "IRDFT node was created with invalid signal size.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op signal size input must be 1D tensor.");
    }

    try {
        auto signal_size = op::Constant::create(element::i64, Shape{2}, {0, 1});
        auto irdft = std::make_shared<op::v9::IRDFT>(data, axes, signal_size);
        FAIL() << "IRDFT node was created with invalid signal size.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Sizes of inputs 'axes' and 'signal_size' of (I)RDFT op must be equal.");
    }
}
