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

struct RDFTConstantAxesAndConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
};

struct RDFTConstantAxesAndConstantSignalSizeTest
    : ::testing::TestWithParam<RDFTConstantAxesAndConstantSignalSizeTestParams> {};

TEST_P(RDFTConstantAxesAndConstantSignalSizeTest, rdft_constant_axes_and_signal_size) {
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

    std::shared_ptr<op::v9::RDFT> rdft;
    if (params.signal_size.empty()) {
        rdft = std::make_shared<op::v9::RDFT>(data, axes_input);
    } else {
        auto signal_size_input =
            op::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
        rdft = std::make_shared<op::v9::RDFT>(data, axes_input, signal_size_input);
    }

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    ASSERT_TRUE(rdft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    RDFTConstantAxesAndConstantSignalSizeTest,
    ::testing::Values(
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180}, {2}, Shape{}, {2, 180, 91, 2}, {1, 2}, {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{6, 180, 180}, {2}, Shape{}, {4, 180, 180, 2}, {2, 0}, {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{16, 500, 180, 369},
                                                        {3},
                                                        Shape{},
                                                        {16, 251, 180, 369, 2},
                                                        {0, 3, 1},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(1, 18)},
                                                        {2},
                                                        Shape{},
                                                        {2, 180, Dimension(1, 10), 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500)},
                                                        {2},
                                                        Shape{},
                                                        {2, 180, Dimension(4, 251), 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180},
                                                        {2},
                                                        Shape{},
                                                        {2, Dimension(7, 500), 91, 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500)},
                                                        {2},
                                                        Shape{},
                                                        {2, Dimension(7, 500), Dimension(4, 251), 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180},
                                                        {2},
                                                        Shape{},
                                                        {Dimension(0, 2), 180, 91, 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500)},
                                                        {2},
                                                        Shape{},
                                                        {Dimension(0, 2), 180, Dimension(4, 251), 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180},
                                                        {2},
                                                        Shape{},
                                                        {Dimension(0, 2), Dimension(7, 500), 91, 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500)},
                                                        {2},
                                                        Shape{},
                                                        {Dimension(0, 2), Dimension(7, 500), Dimension(4, 251), 2},
                                                        {1, 2},
                                                        {}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180}, {2}, {2}, {2, 180, 39, 2}, {1, 2}, {-1, 77}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180}, {2}, {2}, {44, 180, 390, 2}, {2, 0}, {390, 87}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{7, 50, 130, 400},
                                                        {3},
                                                        {3},
                                                        {7, 21, 130, 600, 2},
                                                        {3, 0, 1},
                                                        {600, -1, 40}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180},
                                                        {2},
                                                        {2},
                                                        {2, Dimension(0, 200), 39, 2},
                                                        {1, 2},
                                                        {-1, 77}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400)},
                                                        {2},
                                                        {2},
                                                        {44, 180, 390, 2},
                                                        {2, 0},
                                                        {390, 87}},
        RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500)},
                                                        {3},
                                                        {3},
                                                        {Dimension(8, 129), 21, 130, 600, 2},
                                                        {3, 0, 1},
                                                        {600, -1, 40}}),
    PrintToDummyParamName());

TEST(type_prop, rdft_dynamic_axes) {
    const auto input_shape = PartialShape{2, 180, 180};
    const auto axes_shape = PartialShape::dynamic();
    const auto ref_output_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2};

    auto data = std::make_shared<op::Parameter>(element::f32, input_shape);
    auto axes_input = std::make_shared<op::Parameter>(element::i64, axes_shape);
    auto rdft = std::make_shared<op::v9::RDFT>(data, axes_input);

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    ASSERT_TRUE(rdft->get_output_partial_shape(0).same_scheme(ref_output_shape));
}

struct RDFTNonConstantAxesTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    PartialShape ref_output_shape;
};

struct RDFTNonConstantAxesTest : ::testing::TestWithParam<RDFTNonConstantAxesTestParams> {};

TEST_P(RDFTNonConstantAxesTest, rdft_non_constant_axes) {
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = std::make_shared<op::Parameter>(element::i64, params.axes_shape);
    auto rdft = std::make_shared<op::v9::RDFT>(data, axes_input);

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    ASSERT_TRUE(rdft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    RDFTNonConstantAxesTest,
    ::testing::Values(
        RDFTNonConstantAxesTestParams{{2, 180, 180},
                                      {2},
                                      {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        RDFTNonConstantAxesTestParams{{2, 180, Dimension(7, 500)},
                                      {2},
                                      {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        RDFTNonConstantAxesTestParams{{2, Dimension(7, 500), 180},
                                      {2},
                                      {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        RDFTNonConstantAxesTestParams{{2, Dimension(7, 500), Dimension(7, 500)},
                                      {2},
                                      {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        RDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, 180},
                                      {2},
                                      {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        RDFTNonConstantAxesTestParams{{Dimension(0, 2), 180, Dimension(7, 500)},
                                      {2},
                                      {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        RDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), 180},
                                      {2},
                                      {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        RDFTNonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500)},
                                      {2},
                                      {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}}),
    PrintToDummyParamName());

struct RDFTNonConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
};

struct RDFTNonConstantSignalSizeTest : ::testing::TestWithParam<RDFTNonConstantSignalSizeTestParams> {};

TEST_P(RDFTNonConstantSignalSizeTest, rdft_non_constant_signal_size) {
    auto params = GetParam();

    auto data = std::make_shared<op::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
    auto signal_size_input = std::make_shared<op::Parameter>(element::i64, params.signal_size_shape);
    auto rdft = std::make_shared<op::v9::RDFT>(data, axes_input, signal_size_input);

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    ASSERT_TRUE(rdft->get_output_partial_shape(0).same_scheme(params.ref_output_shape));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    RDFTNonConstantSignalSizeTest,
    ::testing::Values(RDFTNonConstantSignalSizeTestParams{{2, Dimension(0, 200), 180},
                                                          {2},
                                                          {2},
                                                          {2, Dimension(0, 200), Dimension::dynamic(), 2},
                                                          {1, 2}},
                      RDFTNonConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400)},
                                                          {2},
                                                          {2},
                                                          {Dimension::dynamic(), 180, Dimension(0, 400), 2},
                                                          {2, 0}},
                      RDFTNonConstantSignalSizeTestParams{
                          {Dimension(8, 129), 50, 130, Dimension(0, 500)},
                          {3},
                          {3},
                          {Dimension(8, 129), Dimension::dynamic(), 130, Dimension(0, 500), 2},
                          {3, 0, 1}}),
    PrintToDummyParamName());

TEST(type_prop, rdft_invalid_input) {
    auto axes = op::Constant::create(element::i64, Shape{2}, {0, 1});

    try {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{});
        auto rdft = std::make_shared<op::v9::RDFT>(data, axes);
        FAIL() << "RDFT node was created with invalid input.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "The input rank must be greater or equal to 1.");
    }

    try {
        auto data = std::make_shared<op::Parameter>(element::f32, Shape{4});
        auto rdft = std::make_shared<op::v9::RDFT>(data, axes);
        FAIL() << "RDFT node was created with invalid input.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "The input rank must be greater than or equal to the number of RDFT op axes.");
    }
}

TEST(type_prop, rdft_invalid_axes) {
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});

    try {
        auto axes = op::Constant::create(element::i64, Shape{1}, {3});
        auto rdft = std::make_shared<op::v9::RDFT>(data, axes);
        FAIL() << "RDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axis must be less than input rank.");
    }

    try {
        auto axes = op::Constant::create(element::i64, Shape{1}, {-4});
        auto rdft = std::make_shared<op::v9::RDFT>(data, axes);
        FAIL() << "RDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axis must be positive or equal to zero.");
    }

    try {
        auto axes = op::Constant::create(element::i64, Shape{2}, {0, 0});
        auto rdft = std::make_shared<op::v9::RDFT>(data, axes);
        FAIL() << "RDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axes must be unique.");
    }

    try {
        auto axes = op::Constant::create(element::i64, Shape{1, 2}, {0, 1});
        auto rdft = std::make_shared<op::v9::RDFT>(data, axes);
        FAIL() << "RDFT node was created with invalid axes.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op axes input must be 1D tensor.");
    }
}

TEST(type_prop, rdft_invalid_signal_size) {
    auto data = std::make_shared<op::Parameter>(element::f32, Shape{4, 3, 2});
    auto axes = op::Constant::create(element::i64, Shape{1}, {0});

    try {
        auto signal_size = op::Constant::create(element::i64, Shape{1, 2}, {0, 1});
        auto rdft = std::make_shared<op::v9::RDFT>(data, axes, signal_size);
        FAIL() << "RDFT node was created with invalid signal size.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "(I)RDFT op signal size input must be 1D tensor.");
    }

    try {
        auto signal_size = op::Constant::create(element::i64, Shape{2}, {0, 1});
        auto rdft = std::make_shared<op::v9::RDFT>(data, axes, signal_size);
        FAIL() << "RDFT node was created with invalid signal size.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Sizes of inputs 'axes' and 'signal_size' of (I)RDFT op must be equal.");
    }
}
