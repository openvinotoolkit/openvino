// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <type_traits>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gmock/gmock.h"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset12.hpp"

namespace fft_base_test {
using namespace std;
using namespace ov;
using namespace op;
using namespace testing;
using FFTBaseTypes = Types<op::v7::DFT, op::v7::IDFT>;

struct FFTConstantAxesAndConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<int64_t> signal_size;
    std::vector<size_t> expected_symbols;
};

namespace {
TensorSymbol from_idx_to_symbol_vector(const std::vector<size_t>& indices,
                                       const TensorSymbol& initial_symbols,
                                       size_t base = 10) {
    TensorSymbol result;
    for (const auto& i : indices) {
        if (i == 0)
            result.push_back(nullptr);
        else
            result.push_back(initial_symbols[(i >= base ? i - base : i)]);
    }
    return result;
}
}  // namespace

template <class TOp>
class FFTConstantAxesAndConstantSignalSizeTest : public TypePropOpTest<TOp> {
public:
    const std::vector<FFTConstantAxesAndConstantSignalSizeTestParams> test_params{
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, Shape{}, {2, 180, 180, 2}, {1, 2}, {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2}, {2}, Shape{}, {2, 180, 180, 2}, {2, 0}, {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{16, 500, 180, 369, 2},
                                                       {3},
                                                       Shape{},
                                                       {16, 500, 180, 369, 2},
                                                       {0, 3, 1},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {2, 180, 180, Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), 2},
                                                       {2},
                                                       Shape{},
                                                       {2, 180, Dimension(7, 500), 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {2, 180, Dimension(7, 500), Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, 2},
                                                       {2},
                                                       Shape{},
                                                       {2, Dimension(7, 500), 180, 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {2, Dimension(7, 500), 180, Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
                                                       {2},
                                                       Shape{},
                                                       {2, Dimension(7, 500), Dimension(7, 500), 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, 2},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), 180, 180, 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), 180, Dimension(7, 500), 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), Dimension(7, 500), 180, 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                                       {2},
                                                       Shape{},
                                                       {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                                       {1, 2},
                                                       {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            Shape{},
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {1, 2},
            {}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2},
                                                       {2},
                                                       {2},
                                                       {2, 180, 77, 2},
                                                       {1, 2},
                                                       {-1, 77},
                                                       {10, 11, 0, 13}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2},
                                                       {2},
                                                       {2},
                                                       {87, 180, 390, 2},
                                                       {2, 0},
                                                       {390, 87},
                                                       {0, 11, 0, 13}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{7, 50, 130, 400, 2},
                                                       {3},
                                                       {3},
                                                       {7, 40, 130, 600, 2},
                                                       {3, 0, 1},
                                                       {600, -1, 40},
                                                       {10, 0, 12, 0, 14}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                                       {2},
                                                       {2},
                                                       {2, Dimension(0, 200), 77, 2},
                                                       {1, 2},
                                                       {-1, 77},
                                                       {10, 11, 0, 13}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                                       {2},
                                                       {2},
                                                       {87, 180, 390, 2},
                                                       {2, 0},
                                                       {390, 87},
                                                       {0, 11, 0, 13}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                                                       {3},
                                                       {3},
                                                       {Dimension(8, 129), 40, 130, 600, 2},
                                                       {3, 0, 1},
                                                       {600, -1, 40},
                                                       {10, 0, 12, 0, 14}}};
};

TYPED_TEST_SUITE_P(FFTConstantAxesAndConstantSignalSizeTest);

TYPED_TEST_P(FFTConstantAxesAndConstantSignalSizeTest, constant_axes_and_signal_size) {
    for (auto params : this->test_params) {
        auto symbols = set_shape_symbols(params.input_shape);
        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = op::v0::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);

        std::shared_ptr<TypeParam> dft;
        if (params.signal_size.empty()) {
            dft = this->make_op(data, axes_input);
        } else {
            auto signal_size_input =
                op::v0::Constant::create<int64_t>(element::i64, params.signal_size_shape, params.signal_size);
            dft = this->make_op(data, axes_input, signal_size_input);
        }

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);
        if (params.expected_symbols.empty()) {
            EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)), get_shape_symbols(params.input_shape));
        } else {
            auto output_symbols = from_idx_to_symbol_vector(params.expected_symbols, symbols);
            EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)), output_symbols);
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(FFTConstantAxesAndConstantSignalSizeTest, constant_axes_and_signal_size);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, FFTConstantAxesAndConstantSignalSizeTest, FFTBaseTypes);

struct NonConstantAxesTestParams {
    PartialShape input_shape;
    PartialShape axes_shape;
    PartialShape ref_output_shape;
};

template <class TOp>
class FFTNonConstantAxesTest : public TypePropOpTest<TOp> {
public:
    const std::vector<NonConstantAxesTestParams> test_params{
        NonConstantAxesTestParams{{2, 180, 180, Dimension(1, 18)},
                                  PartialShape::dynamic(),
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{1, 180, Dimension(1, 18)},
                                  PartialShape::dynamic(),
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{2, 180, 180, Dimension(1, 18)},
                                  {-1},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{2, 180, 180, Dimension(1, 18)},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{2, 180, Dimension(7, 500), 2},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{{2, 180, Dimension(7, 500), Dimension(1, 18)},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{2, Dimension(7, 500), 180, 2},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{{2, Dimension(7, 500), 180, Dimension(1, 18)},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{2, Dimension(7, 500), Dimension(7, 500), 2},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{{2, Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{Dimension(0, 2), 180, 180, 2},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{{Dimension(0, 2), 180, 180, Dimension(1, 18)},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{Dimension(0, 2), 180, Dimension(7, 500), 2},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{{Dimension(0, 2), 180, Dimension(7, 500), Dimension(1, 18)},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), 180, 2},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), 180, Dimension(1, 18)},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}},
        NonConstantAxesTestParams{{Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), 2},
                                  {2},
                                  {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2}},
        NonConstantAxesTestParams{
            {Dimension(0, 2), Dimension(7, 500), Dimension(7, 500), Dimension(1, 18)},
            {2},
            {Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension(1, 18)}}};
};

TYPED_TEST_SUITE_P(FFTNonConstantAxesTest);

TYPED_TEST_P(FFTNonConstantAxesTest, non_constant_axes_no_signal_size) {
    for (auto params : this->test_params) {
        set_shape_symbols(params.input_shape);

        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, params.axes_shape);

        auto dft = this->make_op(data, axes_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.input_shape);
        EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)), get_shape_symbols(params.input_shape));
    }
}

TYPED_TEST_P(FFTNonConstantAxesTest, non_constant_axes_param_signal_size) {
    for (auto params : this->test_params) {
        set_shape_symbols(params.input_shape);

        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, params.axes_shape);
        auto signal_size_input = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2});

        auto dft = this->make_op(data, axes_input, signal_size_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

        TensorSymbol expected_symbols(params.input_shape.size() - 1, nullptr);
        expected_symbols.push_back(get_shape_symbols(params.input_shape).back());

        EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)), expected_symbols);
    }
}

TYPED_TEST_P(FFTNonConstantAxesTest, non_constant_axes_const_signal_size) {
    for (auto params : this->test_params) {
        set_shape_symbols(params.input_shape);

        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, params.axes_shape);
        auto signal_size_input = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {100, 200});

        auto dft = this->make_op(data, axes_input, signal_size_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

        TensorSymbol expected_symbols(params.input_shape.size() - 1, nullptr);
        expected_symbols.push_back(get_shape_symbols(params.input_shape).back());
        EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)), expected_symbols);
    }
}

REGISTER_TYPED_TEST_SUITE_P(FFTNonConstantAxesTest,
                            non_constant_axes_no_signal_size,
                            non_constant_axes_param_signal_size,
                            non_constant_axes_const_signal_size);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, FFTNonConstantAxesTest, FFTBaseTypes);

struct NonConstantSignalSizeTestParams {
    PartialShape input_shape;
    Shape axes_shape;
    Shape signal_size_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    std::vector<size_t> expected_symbols;
};

template <class TOp>
class FFTNonConstantSignalSizeTest : public TypePropOpTest<TOp> {
public:
    const std::vector<NonConstantSignalSizeTestParams> test_params{
        NonConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                        {2},
                                        {2},
                                        {2, Dimension::dynamic(), Dimension::dynamic(), 2},
                                        {1, 2},
                                        {10, 0, 0, 13}},
        NonConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                        {2},
                                        {2},
                                        {Dimension::dynamic(), 180, Dimension::dynamic(), 2},
                                        {2, 0},
                                        {0, 11, 0, 13}},
        NonConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                                        {3},
                                        {3},
                                        {Dimension::dynamic(), Dimension::dynamic(), 130, Dimension::dynamic(), 2},
                                        {3, 0, 1},
                                        {0, 0, 12, 0, 14}}};
};

TYPED_TEST_SUITE_P(FFTNonConstantSignalSizeTest);

TYPED_TEST_P(FFTNonConstantSignalSizeTest, non_constant_signal_size) {
    for (auto params : this->test_params) {
        auto symbols = set_shape_symbols(params.input_shape);
        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = op::v0::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
        auto signal_size_input = std::make_shared<op::v0::Parameter>(element::i64, params.signal_size_shape);
        auto dft = this->make_op(data, axes_input, signal_size_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);
        auto output_symbols = from_idx_to_symbol_vector(params.expected_symbols, symbols);
        EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)), output_symbols);
    }
}

REGISTER_TYPED_TEST_SUITE_P(FFTNonConstantSignalSizeTest, non_constant_signal_size);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, FFTNonConstantSignalSizeTest, FFTBaseTypes);

template <class TOp>
class FFTInvalidInput : public TypePropOpTest<TOp> {};

TYPED_TEST_SUITE_P(FFTInvalidInput);

TYPED_TEST_P(FFTInvalidInput, invalid_input_data) {
    auto axes = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});

    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{2});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("The input rank must be greater or equal to 2"));

    data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("The last dimension of input data must be 2"));

    data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 2});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("The input rank must be greater than number of axes."));
}

TYPED_TEST_P(FFTInvalidInput, invalid_axes) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 2});

    auto axes = op::v0::Constant::create(element::i64, Shape{1}, {3});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("Axis 3 out of the tensor rank range [-2, 1]"));

    axes = op::v0::Constant::create(element::i64, Shape{1}, {-3});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("Axis -3 out of the tensor rank range [-2, 1]"));

    axes = op::v0::Constant::create(element::i64, Shape{2}, {0, -2});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes), ov::Exception, HasSubstr("Each axis must be unique"));

    axes = op::v0::Constant::create(element::i64, Shape{1}, {2});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("Axis 2 out of the tensor rank range [-2, 1]"));

    axes = op::v0::Constant::create(element::i64, Shape{1, 2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes), ov::Exception, HasSubstr("Axes input must be 1D tensor."));
}

TYPED_TEST_P(FFTInvalidInput, invalid_signal_size) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 2});
    auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});

    auto signal_size = op::v0::Constant::create(element::i64, Shape{1, 2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes, signal_size),
                    ov::Exception,
                    HasSubstr("Signal size input must be 1D tensor"));

    signal_size = op::v0::Constant::create(element::i64, Shape{2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes, signal_size),
                    ov::Exception,
                    HasSubstr("Sizes of inputs 'axes' and 'signal_size' must be equal."));
}

REGISTER_TYPED_TEST_SUITE_P(FFTInvalidInput, invalid_input_data, invalid_axes, invalid_signal_size);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, FFTInvalidInput, FFTBaseTypes);

template <class TOp>
class FFTDynamicTypes : public TypePropOpTest<TOp> {};

TYPED_TEST_SUITE_P(FFTDynamicTypes);

TYPED_TEST_P(FFTDynamicTypes, dynamic_types) {
    const auto input_shape = PartialShape{2, 180, 180, 2};
    const auto axes_shape = PartialShape::dynamic();
    const auto signal_size_shape = PartialShape::dynamic();
    const auto ref_output_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 2};

    auto data = std::make_shared<op::v0::Parameter>(element::dynamic, input_shape);
    auto axes_input = std::make_shared<op::v0::Parameter>(element::dynamic, axes_shape);
    auto signal_size_input = std::make_shared<op::v0::Parameter>(element::dynamic, signal_size_shape);
    auto dft = this->make_op(data, axes_input, signal_size_input);

    EXPECT_EQ(dft->get_element_type(), element::dynamic);
    EXPECT_EQ(dft->get_output_partial_shape(0), ref_output_shape);
}

REGISTER_TYPED_TEST_SUITE_P(FFTDynamicTypes, dynamic_types);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, FFTDynamicTypes, FFTBaseTypes);

struct FFTConstantAxesAndShapeOfSignalSizeTestParams {
    PartialShape input_shape;
    PartialShape ref_output_shape;
    std::vector<int64_t> axes;
    PartialShape signal_size;
    std::vector<size_t> expected_symbols;
};

template <class TOp>
class FFTConstantAxesAndShapeOfSignalSizeTest : public TypePropOpTest<TOp> {
public:
    const std::vector<FFTConstantAxesAndShapeOfSignalSizeTestParams> test_params{
        FFTConstantAxesAndShapeOfSignalSizeTestParams{{3, {10, 16}, 18, 20, 2},
                                                      {-1, {10, 16}, {6, 8}, 4, 2},
                                                      {2, 0, -1},
                                                      {{6, 8}, -1, 4},
                                                      {21, 11, 20, 22, 14}},
        FFTConstantAxesAndShapeOfSignalSizeTestParams{{{8, 129}, 50, 130, {0, 500}, 2},
                                                      {{8, 129}, {0, 10}, -1, 40, 2},
                                                      {1, 2, 3},
                                                      {{0, 10}, -1, 40},
                                                      {10, 20, 21, 22, 14}}};
};
TYPED_TEST_SUITE_P(FFTConstantAxesAndShapeOfSignalSizeTest);

TYPED_TEST_P(FFTConstantAxesAndShapeOfSignalSizeTest, constant_axes_and_shape_of_signal_size) {
    for (auto params : this->test_params) {
        auto in_symbols = set_shape_symbols(params.input_shape);
        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = op::v0::Constant::create<int64_t>(element::i64, Shape{params.axes.size()}, params.axes);

        auto signal_symbols = set_shape_symbols(params.signal_size);
        auto param_of_shape = std::make_shared<op::v0::Parameter>(element::f32, params.signal_size);
        auto signal_size_input = std::make_shared<op::v3::ShapeOf>(param_of_shape, element::i64);

        auto dft = this->make_op(data, axes_input, signal_size_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);
        TensorSymbol expected_output_symbols;
        for (const auto& i : params.expected_symbols) {
            if (i >= 20) {
                expected_output_symbols.push_back(signal_symbols[i - 20]);
            } else if (i >= 10) {
                expected_output_symbols.push_back(in_symbols[i - 10]);
            }
        }
        EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)), expected_output_symbols);
    }
}

TYPED_TEST_P(FFTConstantAxesAndShapeOfSignalSizeTest, constant_axes_and_shape_of_signal_size_first_input_symbols) {
    auto params = FFTConstantAxesAndShapeOfSignalSizeTestParams{{3, {10, 16}, 18, 20, 2},
                                                                {-1, {10, 16}, {6, 8}, 4, 2},
                                                                {2, 0, -1},
                                                                {{6, 8}, -1, 4},
                                                                {0, 11, 0, 0, 14}};
    auto symbols = set_shape_symbols(params.input_shape);
    auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, Shape{params.axes.size()}, params.axes);

    auto param_of_shape = std::make_shared<op::v0::Parameter>(element::f32, params.signal_size);
    auto signal_size_input = std::make_shared<op::v3::ShapeOf>(param_of_shape, element::i64);

    auto dft = this->make_op(data, axes_input, signal_size_input);

    EXPECT_EQ(dft->get_element_type(), element::f32);
    EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

    EXPECT_THAT(get_shape_symbols(dft->get_output_partial_shape(0)),
                ElementsAre(nullptr, symbols[1], nullptr, nullptr, symbols[4]));
}

TYPED_TEST_P(FFTConstantAxesAndShapeOfSignalSizeTest, constant_axes_and_shape_of_signal_size_second_input_symbols) {
    auto params = FFTConstantAxesAndShapeOfSignalSizeTestParams{{3, {10, 16}, 18, 20, 2},
                                                                {-1, {10, 16}, {6, 8}, 4, 2},
                                                                {2, 0, -1},
                                                                {{6, 8}, -1, 4},
                                                                {21, 0, 20, 22, 0}};

    auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, Shape{params.axes.size()}, params.axes);

    auto symbols = set_shape_symbols(params.signal_size);
    auto param_of_shape = std::make_shared<op::v0::Parameter>(element::f32, params.signal_size);
    auto signal_size_input = std::make_shared<op::v3::ShapeOf>(param_of_shape, element::i64);

    auto dft = this->make_op(data, axes_input, signal_size_input);

    EXPECT_EQ(dft->get_element_type(), element::f32);
    EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

    EXPECT_THAT(get_shape_symbols(dft->get_output_partial_shape(0)),
                ElementsAre(symbols[1], nullptr, symbols[0], symbols[2], nullptr));
}

TYPED_TEST_P(FFTConstantAxesAndShapeOfSignalSizeTest, constant_axes_and_shape_of_signal_size_single_symbol) {
    auto params = FFTConstantAxesAndShapeOfSignalSizeTestParams{{3, {10, 16}, 18, 20, 2},
                                                                {-1, {10, 16}, {6, 8}, 4, 2},
                                                                {2, 0, -1},
                                                                {{6, 8}, -1, 4},
                                                                {0, 11, 22, 0, 0}};
    auto A = make_shared<Symbol>(), B = make_shared<Symbol>();
    params.input_shape[1].set_symbol(A);

    auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, Shape{params.axes.size()}, params.axes);

    params.signal_size[0].set_symbol(B);

    auto param_of_shape = std::make_shared<op::v0::Parameter>(element::f32, params.signal_size);
    auto signal_size_input = std::make_shared<op::v3::ShapeOf>(param_of_shape, element::i64);

    auto dft = this->make_op(data, axes_input, signal_size_input);

    EXPECT_EQ(dft->get_element_type(), element::f32);
    EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

    EXPECT_THAT(get_shape_symbols(dft->get_output_partial_shape(0)), ElementsAre(nullptr, A, B, nullptr, nullptr));
}

TYPED_TEST_P(FFTConstantAxesAndShapeOfSignalSizeTest, constant_axes_and_shape_of_signal_size_no_symbols) {
    auto params = FFTConstantAxesAndShapeOfSignalSizeTestParams{{3, {10, 16}, 18, 20, 2},
                                                                {-1, {10, 16}, {6, 8}, 4, 2},
                                                                {2, 0, -1},
                                                                {{6, 8}, -1, 4},
                                                                {0, 0, 0, 0, 0}};
    auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, Shape{params.axes.size()}, params.axes);

    auto param_of_shape = std::make_shared<op::v0::Parameter>(element::f32, params.signal_size);
    auto signal_size_input = std::make_shared<op::v3::ShapeOf>(param_of_shape, element::i64);

    auto dft = this->make_op(data, axes_input, signal_size_input);

    EXPECT_EQ(dft->get_element_type(), element::f32);
    EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

    EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)),
              TensorSymbol({nullptr, nullptr, nullptr, nullptr, nullptr}));
}

TYPED_TEST_P(FFTConstantAxesAndShapeOfSignalSizeTest, constant_axes_and_shape_of_concat_signal_size) {
    auto params = FFTConstantAxesAndShapeOfSignalSizeTestParams{{4, {8, 16}, 24, -1, 2},
                                                                {{5, 10}, {8, 16}, -1, 40, 2},
                                                                {1, 0, -2, 3},
                                                                {{5, 10}, -1, 40}};
    auto input_symbols = set_shape_symbols(params.input_shape);
    auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
    auto axes_input = op::v0::Constant::create<int64_t>(element::i64, Shape{params.axes.size()}, params.axes);

    auto signal_symbols = set_shape_symbols(params.signal_size);
    auto param_of_shape = std::make_shared<op::v0::Parameter>(element::f32, params.signal_size);
    auto shape_of = std::make_shared<op::v3::ShapeOf>(param_of_shape, element::i64);
    auto minus_one = op::v0::Constant::create<int64_t>(element::i64, Shape{1}, {-1});

    auto signal_size_input = std::make_shared<op::v0::Concat>(OutputVector{minus_one, shape_of}, 0);

    auto dft = this->make_op(data, axes_input, signal_size_input);

    EXPECT_EQ(dft->get_element_type(), element::f32);
    EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);
    auto output_symbols =
        TensorSymbol{signal_symbols[0], input_symbols[1], signal_symbols[1], signal_symbols[2], input_symbols[4]};
    EXPECT_EQ(get_shape_symbols(dft->get_output_partial_shape(0)), output_symbols);
}

REGISTER_TYPED_TEST_SUITE_P(FFTConstantAxesAndShapeOfSignalSizeTest,
                            constant_axes_and_shape_of_signal_size,
                            constant_axes_and_shape_of_signal_size_first_input_symbols,
                            constant_axes_and_shape_of_signal_size_second_input_symbols,
                            constant_axes_and_shape_of_signal_size_single_symbol,
                            constant_axes_and_shape_of_signal_size_no_symbols,
                            constant_axes_and_shape_of_concat_signal_size);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, FFTConstantAxesAndShapeOfSignalSizeTest, FFTBaseTypes);

}  // namespace fft_base_test
