// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rdft.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;
using namespace testing;

struct RDFTConstantAxesAndConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    std::vector<size_t> expected_symbols;
};

namespace {
TensorSymbol from_idx_to_symbol_vector(const std::vector<size_t>& indices, const TensorSymbol& initial_symbols) {
    TensorSymbol result;
    for (const auto& i : indices) {
        if (i == 0)
            result.push_back(nullptr);
        else
            result.push_back(initial_symbols[i - 10]);
    }
    return result;
}
}  // namespace

struct RDFTConstantAxesAndConstantSignalSizeTest
    : ::testing::TestWithParam<RDFTConstantAxesAndConstantSignalSizeTestParams> {};

TEST_P(RDFTConstantAxesAndConstantSignalSizeTest, rdft_constant_axes_and_signal_size) {
    auto params = GetParam();

    auto input_shape = params.input_shape;
    auto symbols = set_shape_symbols(input_shape);
    auto data = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

    std::shared_ptr<op::v9::RDFT> rdft;
    if (params.signal_size.empty()) {
        rdft = std::make_shared<ov::op::v9::RDFT>(data, axes_input);
    } else {
        auto signal_size_input =
            op::v0::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
        rdft = std::make_shared<op::v9::RDFT>(data, axes_input, signal_size_input);
    }

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    EXPECT_EQ(rdft->get_output_partial_shape(0), params.ref_output_shape);
    auto output_expected_symbols = from_idx_to_symbol_vector(params.expected_symbols, symbols);
    EXPECT_EQ(get_shape_symbols(rdft->get_output_partial_shape(0)), output_expected_symbols);
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    RDFTConstantAxesAndConstantSignalSizeTest,
    ::testing::Values(RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180},
                                                                      {2},
                                                                      Shape{},
                                                                      {2, 180, 91, 2},
                                                                      {1, 2},
                                                                      {},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{6, 180, 180},
                                                                      {2},
                                                                      Shape{},
                                                                      {4, 180, 180, 2},
                                                                      {2, 0},
                                                                      {},
                                                                      {0, 11, 12, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{16, 500, 180, 369},
                                                                      {3},
                                                                      Shape{},
                                                                      {16, 251, 180, 369, 2},
                                                                      {0, 3, 1},
                                                                      {},
                                                                      {10, 0, 12, 13, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(1, 18)},
                                                                      {2},
                                                                      Shape{},
                                                                      {2, 180, Dimension(1, 10), 2},
                                                                      {1, 2},
                                                                      {},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500)},
                                                                      {2},
                                                                      Shape{},
                                                                      {2, 180, Dimension(4, 251), 2},
                                                                      {1, 2},
                                                                      {},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180},
                                                                      {2},
                                                                      Shape{},
                                                                      {2, Dimension(7, 500), 91, 2},
                                                                      {1, 2},
                                                                      {},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500)},
                                                                      {2},
                                                                      Shape{},
                                                                      {2, Dimension(7, 500), Dimension(4, 251), 2},
                                                                      {1, 2},
                                                                      {},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180},
                                                                      {2},
                                                                      Shape{},
                                                                      {Dimension(0, 2), 180, 91, 2},
                                                                      {1, 2},
                                                                      {},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500)},
                                                                      {2},
                                                                      Shape{},
                                                                      {Dimension(0, 2), 180, Dimension(4, 251), 2},
                                                                      {1, 2},
                                                                      {},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180},
                                                                      {2},
                                                                      Shape{},
                                                                      {Dimension(0, 2), Dimension(7, 500), 91, 2},
                                                                      {1, 2},
                                                                      {},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{
                          {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500)},
                          {2},
                          Shape{},
                          {Dimension(0, 2), Dimension(7, 500), Dimension(4, 251), 2},
                          {1, 2},
                          {},
                          {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180},
                                                                      {2},
                                                                      {2},
                                                                      {2, 180, 39, 2},
                                                                      {1, 2},
                                                                      {-1, 77},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180},
                                                                      {2},
                                                                      {2},
                                                                      {44, 180, 390, 2},
                                                                      {2, 0},
                                                                      {390, 87},
                                                                      {0, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{7, 50, 130, 400},
                                                                      {3},
                                                                      {3},
                                                                      {7, 21, 130, 600, 2},
                                                                      {3, 0, 1},
                                                                      {600, -1, 40},
                                                                      {10, 0, 12, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180},
                                                                      {2},
                                                                      {2},
                                                                      {2, Dimension(0, 200), 39, 2},
                                                                      {1, 2},
                                                                      {-1, 77},
                                                                      {10, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400)},
                                                                      {2},
                                                                      {2},
                                                                      {44, 180, 390, 2},
                                                                      {2, 0},
                                                                      {390, 87},
                                                                      {0, 11, 0, 0}},
                      RDFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500)},
                                                                      {3},
                                                                      {3},
                                                                      {Dimension(8, 129), 21, 130, 600, 2},
                                                                      {3, 0, 1},
                                                                      {600, -1, 40},
                                                                      {10, 0, 12, 0, 0}}),
    PrintToDummyParamName());

TEST(type_prop, rdft_dynamic_axes) {
    const auto input_shape = PartialShape{2, 180, 180};
    const auto axes_shape = PartialShape::dynamic();
    const auto ref_output_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2};

    auto data = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, axes_shape);
    auto rdft = std::make_shared<op::v9::RDFT>(data, axes_input);

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    EXPECT_EQ(rdft->get_output_partial_shape(0), ref_output_shape);
}

struct RDFTNonConstantAxesTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    PartialShape ref_output_shape;
};

struct RDFTNonConstantAxesTest : ::testing::TestWithParam<RDFTNonConstantAxesTestParams> {};

TEST_P(RDFTNonConstantAxesTest, rdft_non_constant_axes) {
    auto params = GetParam();

    auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
    auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, params.axes_shape);
    auto rdft = std::make_shared<op::v9::RDFT>(data, axes_input);

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    EXPECT_EQ(rdft->get_output_partial_shape(0), params.ref_output_shape);
    EXPECT_THAT(get_shape_symbols(rdft->get_output_partial_shape(0)), Each(nullptr));
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
    std::vector<size_t> expected_symbols;
};

struct RDFTNonConstantSignalSizeTest : ::testing::TestWithParam<RDFTNonConstantSignalSizeTestParams> {};

TEST_P(RDFTNonConstantSignalSizeTest, rdft_non_constant_signal_size) {
    auto params = GetParam();

    auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
    auto signal_size_input = std::make_shared<op::v0::Parameter>(element::i64, params.signal_size_shape);
    auto rdft = std::make_shared<op::v9::RDFT>(data, axes_input, signal_size_input);

    EXPECT_EQ(rdft->get_element_type(), element::f32);
    EXPECT_EQ(rdft->get_output_partial_shape(0), params.ref_output_shape);
    EXPECT_EQ(get_shape_symbols(rdft->get_output_partial_shape(0)),
              TensorSymbol(params.expected_symbols.size(), nullptr));
}

INSTANTIATE_TEST_SUITE_P(
    type_prop,
    RDFTNonConstantSignalSizeTest,
    ::testing::Values(RDFTNonConstantSignalSizeTestParams{{2, Dimension(0, 200), 180},
                                                          {2},
                                                          {2},
                                                          {2, Dimension(0, 200), Dimension::dynamic(), 2},
                                                          {1, 2},
                                                          {0, 0, 0, 0}},
                      RDFTNonConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400)},
                                                          {2},
                                                          {2},
                                                          {Dimension::dynamic(), 180, Dimension(0, 400), 2},
                                                          {2, 0},
                                                          {0, 0, 0, 0}},
                      RDFTNonConstantSignalSizeTestParams{
                          {Dimension(8, 129), 50, 130, Dimension(0, 500)},
                          {3},
                          {3},
                          {Dimension(8, 129), Dimension::dynamic(), 130, Dimension(0, 500), 2},
                          {3, 0, 1},
                          {0, 0, 0, 0, 0}}),
    PrintToDummyParamName());

TEST(type_prop, rdft_invalid_input) {
    auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});

    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::RDFT>(data, axes),
                    ov::Exception,
                    HasSubstr("The input rank must be greater or equal to 1"));

    data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::RDFT>(data, axes),
                    ov::Exception,
                    HasSubstr("The input rank must be greater than or equal to the number of axes."));
}

TEST(type_prop, rdft_invalid_axes) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 2});

    auto axes = op::v0::Constant::create(element::i64, Shape{1}, {3});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::RDFT>(data, axes),
                    ov::Exception,
                    HasSubstr("Axis 3 out of the tensor rank range [-3, 2]"));

    axes = op::v0::Constant::create(element::i64, Shape{1}, {-4});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::RDFT>(data, axes),
                    ov::Exception,
                    HasSubstr("Axis -4 out of the tensor rank range [-3, 2]"));

    axes = op::v0::Constant::create(element::i64, Shape{2}, {0, -3});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::RDFT>(data, axes),
                    ov::Exception,
                    HasSubstr("Each axis must be unique"));

    axes = op::v0::Constant::create(element::i64, Shape{1, 2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::RDFT>(data, axes),
                    ov::Exception,
                    HasSubstr("Axes input must be 1D tensor."));
}

TEST(type_prop, rdft_invalid_signal_size) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 2});
    auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});

    auto signal_size = op::v0::Constant::create(element::i64, Shape{1, 2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::RDFT>(data, axes, signal_size),
                    ov::Exception,
                    HasSubstr("Signal size input must be 1D tensor."));

    signal_size = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::v9::RDFT>(data, axes, signal_size),
                    ov::Exception,
                    HasSubstr("Sizes of inputs 'axes' and 'signal_size' must be equal."));
}

TEST(type_prop, rdft_dynamic_types) {
    const auto input_shape = PartialShape{2, 180, 180};
    const auto axes_shape = PartialShape::dynamic();
    const auto signal_size_shape = PartialShape::dynamic();
    const auto ref_output_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2};

    auto data = std::make_shared<op::v0::Parameter>(element::dynamic, input_shape);
    auto axes_input = std::make_shared<op::v0::Parameter>(element::dynamic, axes_shape);
    auto signal_size_input = std::make_shared<op::v0::Parameter>(element::dynamic, signal_size_shape);
    auto rdft = std::make_shared<op::v9::RDFT>(data, axes_input, signal_size_input);

    EXPECT_EQ(rdft->get_element_type(), element::dynamic);
    EXPECT_EQ(rdft->get_output_partial_shape(0), ref_output_shape);
}
