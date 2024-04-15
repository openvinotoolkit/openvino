// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;
// using T1 = typename element_type_traits<IT>::value_type;
namespace {

struct ReshapeParams {
    template <class T>
    ReshapeParams(const Shape& input_shape,
                  const Shape& expected_shape,
                  const element::Type& input_type,
                  const element::Type& expected_type,
                  const std::vector<T>& input_value,
                  const std::vector<T>& expected_value,
                  const bool zero_flag) {
        m_input_shape = input_shape;
        m_expected_shape = expected_shape;
        m_input_type = input_type;
        m_expected_type = expected_type;
        m_zero_flag = zero_flag;
        m_input_value = CreateTensor(input_shape, input_type, input_value);
        m_expected_value = CreateTensor(expected_shape, expected_type, expected_value);
    }

    template <class T>
    ReshapeParams(const Shape& input_shape,
                  const Shape& expected_shape,
                  const element::Type& input_type,
                  const element::Type& expected_type,
                  const bool zero_flag,
                  T step,
                  const Shape& extra_shape = Shape{}) {
        m_input_shape = input_shape;
        m_expected_shape = expected_shape;
        m_input_type = input_type;
        m_expected_type = expected_type;
        m_zero_flag = zero_flag;

        std::vector<T> value(shape_size(input_shape));
        std::iota(value.begin(), value.end(), static_cast<T>(step));
        m_input_value = CreateTensor(input_shape, input_type, value);
        m_expected_value = CreateTensor(expected_shape, expected_type, value);

        if (extra_shape.size() > 0) {
            m_expected_shape = extra_shape;
        }
    }

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
    bool m_zero_flag;
};

struct ReshapeShuffleParams {
    template <class T>
    ReshapeShuffleParams(const Shape& input_shape1,
                         const Shape& input_shape2,
                         const Shape& input_shape3,
                         const Shape& expected_shape,
                         const element::Type_t& input_type,
                         const element::Type_t& expected_type,
                         const bool zero_flag,
                         T step) {
        m_input_shape1 = input_shape1;
        m_input_shape2 = input_shape2;
        m_input_shape3 = input_shape3;
        m_expected_shape = expected_shape;
        m_input_type = input_type;
        m_expected_type = expected_type;
        m_zero_flag = zero_flag;

        std::vector<T> value(shape_size(input_shape1));
        std::iota(value.begin(), value.end(), static_cast<T>(step));
        m_input_value = CreateTensor(input_shape1, input_type, value);
        m_expected_value = CreateTensor(expected_shape, expected_type, value);
    }

    Shape m_input_shape1;
    Shape m_input_shape2;
    Shape m_input_shape3;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
    bool m_zero_flag;
};

class ReferenceReshapeLayerTest : public testing::TestWithParam<ReshapeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.m_input_type,
                                  params.m_expected_type,
                                  params.m_input_shape,
                                  params.m_expected_shape,
                                  params.m_zero_flag);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReshapeParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "output_type=" << param.m_expected_type;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const element::Type& input_type,
                                                 const element::Type& expected_type,
                                                 const Shape& input_shape,
                                                 const Shape& expected_shape,
                                                 const bool zero_flag) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto reshape = std::make_shared<op::v1::Reshape>(
            in,
            op::v0::Constant::create(element::Type_t::u64, {expected_shape.size()}, expected_shape),
            zero_flag);
        return std::make_shared<Model>(NodeVector{reshape}, ParameterVector{in});
    }
};

class ReferenceReshapeShuffleLayerTest : public testing::TestWithParam<ReshapeShuffleParams>,
                                         public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.m_input_type,
                                  params.m_expected_type,
                                  params.m_input_shape1,
                                  params.m_input_shape2,
                                  params.m_input_shape3,
                                  params.m_expected_shape,
                                  params.m_zero_flag);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReshapeShuffleParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape1 << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "output_type=" << param.m_expected_type;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const element::Type& input_type,
                                                 const element::Type& expected_type,
                                                 const Shape& input_shape1,
                                                 const Shape& input_shape2,
                                                 const Shape& input_shape3,
                                                 const Shape& expected_shape,
                                                 const bool zero_flag) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto reshape1 = std::make_shared<op::v1::Reshape>(
            in,
            op::v0::Constant::create(element::Type_t::u64, {input_shape2.size()}, input_shape2),
            zero_flag);
        const auto reshape2 = std::make_shared<op::v1::Reshape>(
            reshape1,
            op::v0::Constant::create(element::Type_t::u64, {input_shape3.size()}, input_shape3),
            zero_flag);
        const auto reshape3 = std::make_shared<op::v1::Reshape>(
            reshape2,
            op::v0::Constant::create(element::Type_t::u64, {expected_shape.size()}, expected_shape),
            zero_flag);
        return std::make_shared<Model>(NodeVector{reshape3}, ParameterVector{in});
    }
};

TEST_P(ReferenceReshapeLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceReshapeShuffleLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ReshapeParams> generateParamsForReshape() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<ReshapeParams> params{
        ReshapeParams(Shape{2, 2, 3},
                      Shape{12},
                      ET,
                      ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                      false),
        ReshapeParams(Shape{1, 1, 1}, Shape{}, ET, ET, std::vector<T>{6}, std::vector<T>{6}, false),
        ReshapeParams(Shape{}, Shape{1, 1, 1, 1, 1, 1}, ET, ET, std::vector<T>{42}, std::vector<T>{42}, false),
        ReshapeParams(Shape{3}, Shape{3, 1}, ET, ET, std::vector<T>{1, 2, 3}, std::vector<T>{1, 2, 3}, false),
        ReshapeParams(Shape{3}, Shape{1, 3}, ET, ET, std::vector<T>{1, 2, 3}, std::vector<T>{1, 2, 3}, false),
        ReshapeParams(Shape{3}, Shape{1, 3, 1}, ET, ET, std::vector<T>{1, 2, 3}, std::vector<T>{1, 2, 3}, false),
        ReshapeParams(Shape{3, 3},
                      Shape{3, 3},
                      ET,
                      ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      false),
        ReshapeParams(Shape{1}, Shape{}, ET, ET, std::vector<T>{1}, std::vector<T>{1}, false),
        ReshapeParams(Shape{}, Shape{}, ET, ET, std::vector<T>{1}, std::vector<T>{1}, false),
        ReshapeParams(Shape{2, 2, 3, 3, 2, 4}, Shape{3, 2, 2, 4, 3, 2}, ET, ET, false, static_cast<T>(1)),
        ReshapeParams(Shape{2, 2, 5, 5}, Shape{2, 5, 5, 2}, ET, ET, true, static_cast<T>(1), Shape{0, 5, 0, 2})};

    return params;
}

std::vector<ReshapeParams> generateParamsForReshapeString() {
    const auto ET = ov::element::string;
    using T = typename element_type_traits<ov::element::string>::value_type;

    std::vector<ReshapeParams> params{
        ReshapeParams(Shape{2, 2, 3},
                      Shape{12},
                      ET,
                      ET,
                      std::vector<T>{"A", ",b", "c. ", "d D ", " e ", "FgH", "1;2;3;", "\n ", " \n\n ", "\0", " ", "."},
                      std::vector<T>{"A", ",b", "c. ", "d D ", " e ", "FgH", "1;2;3;", "\n ", " \n\n ", "\0", " ", "."},
                      false),
        ReshapeParams(Shape{12},
                      Shape{2, 3, 2},
                      ET,
                      ET,
                      std::vector<T>{"A", ",b", "c. ", "d D ", " e ", "FgH", "1;2;3;", "\n ", " \n\n ", "\0", " ", "."},
                      std::vector<T>{"A", ",b", "c. ", "d D ", " e ", "FgH", "1;2;3;", "\n ", " \n\n ", "\0", " ", "."},
                      false),
        ReshapeParams(Shape{2, 2, 3},
                      Shape{4, 3},
                      ET,
                      ET,
                      std::vector<T>{"A", ",b", "c. ", "d D ", " e ", "FgH", "1;2;3;", "\n ", " \n\n ", "\0", " ", "."},
                      std::vector<T>{"A", ",b", "c. ", "d D ", " e ", "FgH", "1;2;3;", "\n ", " \n\n ", "\0", " ", "."},
                      false),
        ReshapeParams(Shape{4, 3},
                      Shape{2, 3, 2},
                      ET,
                      ET,
                      std::vector<T>{"A", ",b", "c. ", "d D ", " e ", "FgH", "1;2;3;", "\n ", " \n\n ", "\0", " ", "."},
                      std::vector<T>{"A", ",b", "c. ", "d D ", " e ", "FgH", "1;2;3;", "\n ", " \n\n ", "\0", " ", "."},
                      false),
        ReshapeParams(Shape{1},
                      Shape{1, 1},
                      ET,
                      ET,
                      std::vector<T>{" A, a, B; b; "},
                      std::vector<T>{" A, a, B; b; "},
                      false)};
    return params;
}

template <element::Type_t ET>
std::vector<ReshapeParams> generateParamsForReshape8Bit() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<ReshapeParams> params{
        ReshapeParams(Shape{2, 2, 3},
                      Shape{12},
                      ET,
                      ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                      false),
        ReshapeParams(Shape{1, 1, 1}, Shape{}, ET, ET, std::vector<T>{6}, std::vector<T>{6}, false),
        ReshapeParams(Shape{}, Shape{1, 1, 1, 1, 1, 1}, ET, ET, std::vector<T>{42}, std::vector<T>{42}, false),
        ReshapeParams(Shape{3}, Shape{3, 1}, ET, ET, std::vector<T>{1, 2, 3}, std::vector<T>{1, 2, 3}, false),
        ReshapeParams(Shape{3}, Shape{1, 3}, ET, ET, std::vector<T>{1, 2, 3}, std::vector<T>{1, 2, 3}, false),
        ReshapeParams(Shape{3}, Shape{1, 3, 1}, ET, ET, std::vector<T>{1, 2, 3}, std::vector<T>{1, 2, 3}, false),
        ReshapeParams(Shape{1}, Shape{}, ET, ET, std::vector<T>{1}, std::vector<T>{1}, false),
        ReshapeParams(Shape{}, Shape{}, ET, ET, std::vector<T>{1}, std::vector<T>{1}, false)};

    return params;
}

template <element::Type_t ET>
std::vector<ReshapeShuffleParams> generateParamsForReshapeShuffle() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<ReshapeShuffleParams> params{ReshapeShuffleParams(Shape{1, 112, 56, 56},
                                                                  Shape{1, 4, 28, 56, 56},
                                                                  Shape{1, 28, 4, 56, 56},
                                                                  Shape{1, 112, 56, 56},
                                                                  ET,
                                                                  ET,
                                                                  false,
                                                                  static_cast<T>(1))};

    return params;
}

std::vector<ReshapeParams> generateCombinedParamsForReshape() {
    const std::vector<std::vector<ReshapeParams>> allTypeParams{generateParamsForReshape<element::Type_t::f32>(),
                                                                generateParamsForReshape<element::Type_t::i64>(),
                                                                generateParamsForReshape<element::Type_t::i32>(),
                                                                generateParamsForReshape<element::Type_t::i16>(),
                                                                generateParamsForReshape<element::Type_t::u64>(),
                                                                generateParamsForReshape<element::Type_t::u32>(),
                                                                generateParamsForReshape<element::Type_t::u16>(),
                                                                generateParamsForReshape8Bit<element::Type_t::i8>(),
                                                                generateParamsForReshape8Bit<element::Type_t::u8>(),
                                                                generateParamsForReshapeString()};

    std::vector<ReshapeParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<ReshapeShuffleParams> generateCombinedParamsForReshapeShuffle() {
    const std::vector<std::vector<ReshapeShuffleParams>> allTypeParams{
        generateParamsForReshapeShuffle<element::Type_t::f32>(),
        generateParamsForReshapeShuffle<element::Type_t::i64>(),
        generateParamsForReshapeShuffle<element::Type_t::i32>(),
        generateParamsForReshapeShuffle<element::Type_t::i16>(),
        generateParamsForReshapeShuffle<element::Type_t::i8>(),
        generateParamsForReshapeShuffle<element::Type_t::u64>(),
        generateParamsForReshapeShuffle<element::Type_t::u32>(),
        generateParamsForReshapeShuffle<element::Type_t::u16>(),
        generateParamsForReshapeShuffle<element::Type_t::u8>(),
    };

    std::vector<ReshapeShuffleParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Reshape_With_Hardcoded_Refs,
                         ReferenceReshapeLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReshape()),
                         ReferenceReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reshape_Shuffle_With_Hardcoded_Refs,
                         ReferenceReshapeShuffleLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReshapeShuffle()),
                         ReferenceReshapeShuffleLayerTest::getTestCaseName);

}  // namespace
