// Copyright (C) 2018-2023 Intel Corporation
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
    std::vector<ov::label_t> expected_labels;
};

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
                                                       {10, 11, ov::no_label, 13}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, 180, 180, 2},
                                                       {2},
                                                       {2},
                                                       {87, 180, 390, 2},
                                                       {2, 0},
                                                       {390, 87},
                                                       {ov::no_label, 11, ov::no_label, 13}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{7, 50, 130, 400, 2},
                                                       {3},
                                                       {3},
                                                       {7, 40, 130, 600, 2},
                                                       {3, 0, 1},
                                                       {600, -1, 40},
                                                       {10, ov::no_label, 12, ov::no_label, 14}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{2, Dimension(0, 200), 180, 2},
                                                       {2},
                                                       {2},
                                                       {2, Dimension(0, 200), 77, 2},
                                                       {1, 2},
                                                       {-1, 77},
                                                       {10, 11, ov::no_label, 13}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                                       {2},
                                                       {2},
                                                       {87, 180, 390, 2},
                                                       {2, 0},
                                                       {390, 87},
                                                       {ov::no_label, 11, ov::no_label, 13}},
        FFTConstantAxesAndConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                                                       {3},
                                                       {3},
                                                       {Dimension(8, 129), 40, 130, 600, 2},
                                                       {3, 0, 1},
                                                       {600, -1, 40},
                                                       {10, ov::no_label, 12, ov::no_label, 14}}};
};

TYPED_TEST_SUITE_P(FFTConstantAxesAndConstantSignalSizeTest);

TYPED_TEST_P(FFTConstantAxesAndConstantSignalSizeTest, constant_axes_and_signal_size) {
    for (auto params : this->test_params) {
        set_shape_labels(params.input_shape, 10);
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
        if (params.expected_labels.empty()) {
            EXPECT_EQ(get_shape_labels(dft->get_output_partial_shape(0)), get_shape_labels(params.input_shape));
        } else {
            EXPECT_EQ(get_shape_labels(dft->get_output_partial_shape(0)), params.expected_labels);
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
        set_shape_labels(params.input_shape, 10);

        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, params.axes_shape);

        auto dft = this->make_op(data, axes_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

        std::vector<label_t> expected_labels(params.input_shape.size() - 1, no_label);
        expected_labels.push_back(get_shape_labels(params.input_shape).back());
        EXPECT_EQ(get_shape_labels(dft->get_output_partial_shape(0)), expected_labels);
    }
}

TYPED_TEST_P(FFTNonConstantAxesTest, non_constant_axes_param_signal_size) {
    for (auto params : this->test_params) {
        set_shape_labels(params.input_shape, 10);

        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, params.axes_shape);
        auto signal_size_input = std::make_shared<op::v0::Parameter>(element::i64, PartialShape{2});

        auto dft = this->make_op(data, axes_input, signal_size_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

        std::vector<label_t> expected_labels(params.input_shape.size() - 1, no_label);
        expected_labels.push_back(get_shape_labels(params.input_shape).back());
        EXPECT_EQ(get_shape_labels(dft->get_output_partial_shape(0)), expected_labels);
    }
}

TYPED_TEST_P(FFTNonConstantAxesTest, non_constant_axes_const_signal_size) {
    for (auto params : this->test_params) {
        set_shape_labels(params.input_shape, 10);

        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = std::make_shared<op::v0::Parameter>(element::i64, params.axes_shape);
        auto signal_size_input = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {100, 200});

        auto dft = this->make_op(data, axes_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);

        std::vector<label_t> expected_labels(params.input_shape.size() - 1, no_label);
        expected_labels.push_back(get_shape_labels(params.input_shape).back());
        EXPECT_EQ(get_shape_labels(dft->get_output_partial_shape(0)), expected_labels);
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
    std::vector<ov::label_t> expected_labels;
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
                                        {10, no_label, no_label, 13}},
        NonConstantSignalSizeTestParams{{Dimension(0, 18), 180, Dimension(0, 400), 2},
                                        {2},
                                        {2},
                                        {Dimension::dynamic(), 180, Dimension::dynamic(), 2},
                                        {2, 0},
                                        {no_label, 11, no_label, 13}},
        NonConstantSignalSizeTestParams{{Dimension(8, 129), 50, 130, Dimension(0, 500), 2},
                                        {3},
                                        {3},
                                        {Dimension::dynamic(), Dimension::dynamic(), 130, Dimension::dynamic(), 2},
                                        {3, 0, 1},
                                        {no_label, no_label, 12, no_label, 14}}};
};

TYPED_TEST_SUITE_P(FFTNonConstantSignalSizeTest);

TYPED_TEST_P(FFTNonConstantSignalSizeTest, non_constant_signal_size) {
    for (auto params : this->test_params) {
        set_shape_labels(params.input_shape, 10);
        auto data = std::make_shared<op::v0::Parameter>(element::f32, params.input_shape);
        auto axes_input = op::v0::Constant::create<int64_t>(element::i64, params.axes_shape, params.axes);
        auto signal_size_input = std::make_shared<op::v0::Parameter>(element::i64, params.signal_size_shape);
        auto dft = this->make_op(data, axes_input, signal_size_input);

        EXPECT_EQ(dft->get_element_type(), element::f32);
        EXPECT_EQ(dft->get_output_partial_shape(0), params.ref_output_shape);
        EXPECT_EQ(get_shape_labels(dft->get_output_partial_shape(0)), params.expected_labels);
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
                    HasSubstr("The input rank must be greater or equal to 2."));

    data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("The last dimension of input data must be 2."));

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
                    HasSubstr("Axis value: 3, must be in range (-3, 2)"));

    axes = op::v0::Constant::create(element::i64, Shape{1}, {-3});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("Axis value: -3, must be in range (-3, 2)"));

    axes = op::v0::Constant::create(element::i64, Shape{2}, {0, -2});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes), ov::Exception, HasSubstr("Each axis must be unique"));

    axes = op::v0::Constant::create(element::i64, Shape{1}, {2});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes),
                    ov::Exception,
                    HasSubstr("Axis value: 2, must be in range (-3, 2)"));

    axes = op::v0::Constant::create(element::i64, Shape{1, 2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes), ov::Exception, HasSubstr("Axes input must be 1D tensor."));
}

TYPED_TEST_P(FFTInvalidInput, invalid_signal_size) {
    auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 2});
    auto axes = op::v0::Constant::create(element::i64, Shape{1}, {0});

    auto signal_size = op::v0::Constant::create(element::i64, Shape{1, 2}, {0, 1});
    OV_EXPECT_THROW(std::ignore = this->make_op(data, axes, signal_size),
                    ov::Exception,
                    HasSubstr("Signal size input must be 1D tensor."));

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

}  // namespace fft_base_test
