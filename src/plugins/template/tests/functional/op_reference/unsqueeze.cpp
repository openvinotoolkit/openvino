// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unsqueeze.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct UnsqueezeParams {
    template <class IO_T, class Axes_T>
    UnsqueezeParams(const Shape& input_shape,
                    const Shape& expected_shape,
                    const element::Type& input_type,
                    const element::Type& expected_type,
                    const std::vector<IO_T>& input_value,
                    const std::vector<IO_T>& expected_value,
                    const Shape& axes_shape,
                    const element::Type& axes_type,
                    const std::vector<Axes_T>& axes_value)
        : m_input_shape(input_shape),
          m_expected_shape(expected_shape),
          m_input_type(input_type),
          m_expected_type(expected_type),
          m_input_value(CreateTensor(input_shape, input_type, input_value)),
          m_expected_value(CreateTensor(expected_shape, expected_type, expected_value)),
          m_axes_shape(axes_shape),
          m_axes_type(axes_type),
          m_axes_value(CreateTensor(axes_shape, axes_type, axes_value)) {}

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
    Shape m_axes_shape;
    element::Type m_axes_type;
    ov::Tensor m_axes_value;
};

class ReferenceUnsqueezeLayerTest : public testing::TestWithParam<UnsqueezeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params.m_input_type,
                                  params.m_input_shape,
                                  params.m_axes_type,
                                  params.m_axes_shape,
                                  params.m_axes_value);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<UnsqueezeParams>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape << "; ";
        result << "expected_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "expected_type=" << param.m_expected_type << "; ";
        result << "axes_shape=" << param.m_axes_shape << "; ";
        result << "axes_type=" << param.m_axes_type << "; ";

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const element::Type_t& input_type,
                                                 const Shape& input_shape,
                                                 const element::Type& axes_type,
                                                 const Shape& axes_shape,
                                                 const ov::Tensor& axes_value) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto axes = std::make_shared<op::v0::Constant>(axes_type, axes_shape, axes_value.data());
        const auto unsqueeze = std::make_shared<op::v0::Unsqueeze>(in, axes);
        return std::make_shared<ov::Model>(unsqueeze, ParameterVector{in});
    }
};

TEST_P(ReferenceUnsqueezeLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IO_T, element::Type_t Axes_T>
std::vector<UnsqueezeParams> generateParamsForUnsqueeze() {
    using T1 = typename element_type_traits<IO_T>::value_type;
    using T2 = typename element_type_traits<Axes_T>::value_type;

    std::vector<UnsqueezeParams> params{UnsqueezeParams(Shape{4, 2},
                                                        Shape{4, 1, 1, 2},
                                                        IO_T,
                                                        IO_T,
                                                        std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                                                        std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                                                        Shape{2},
                                                        Axes_T,
                                                        std::vector<T2>{1, 2})};

    return params;
}

template <element::Type_t IO_T, element::Type_t Axes_T>
std::vector<UnsqueezeParams> generateParamsForUnsqueezeNegative() {
    using T1 = typename element_type_traits<IO_T>::value_type;
    using T2 = typename element_type_traits<Axes_T>::value_type;

    std::vector<UnsqueezeParams> params{UnsqueezeParams(Shape{4, 2},
                                                        Shape{4, 1, 2, 1},
                                                        IO_T,
                                                        IO_T,
                                                        std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                                                        std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                                                        Shape{2},
                                                        Axes_T,
                                                        std::vector<T2>{1, -1})};

    return params;
}

std::vector<UnsqueezeParams> generateCombinedParamsForUnsqueeze() {
    const std::vector<std::vector<UnsqueezeParams>> allTypeParams{
        generateParamsForUnsqueeze<element::Type_t::f32, element::Type_t::i64>(),
        generateParamsForUnsqueeze<element::Type_t::f16, element::Type_t::i64>(),
        generateParamsForUnsqueeze<element::Type_t::i64, element::Type_t::i64>(),
        generateParamsForUnsqueeze<element::Type_t::i32, element::Type_t::i64>(),
        generateParamsForUnsqueeze<element::Type_t::u64, element::Type_t::i64>(),
        generateParamsForUnsqueeze<element::Type_t::u32, element::Type_t::i64>(),
        generateParamsForUnsqueeze<element::Type_t::f32, element::Type_t::i32>(),
        generateParamsForUnsqueeze<element::Type_t::f16, element::Type_t::i32>(),
        generateParamsForUnsqueeze<element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForUnsqueeze<element::Type_t::i32, element::Type_t::i32>(),
        generateParamsForUnsqueeze<element::Type_t::u64, element::Type_t::i32>(),
        generateParamsForUnsqueeze<element::Type_t::u32, element::Type_t::i32>(),
        generateParamsForUnsqueezeNegative<element::Type_t::f32, element::Type_t::i64>(),
        generateParamsForUnsqueezeNegative<element::Type_t::f16, element::Type_t::i64>(),
        generateParamsForUnsqueezeNegative<element::Type_t::i64, element::Type_t::i64>(),
        generateParamsForUnsqueezeNegative<element::Type_t::i32, element::Type_t::i64>(),
        generateParamsForUnsqueezeNegative<element::Type_t::f32, element::Type_t::i32>(),
        generateParamsForUnsqueezeNegative<element::Type_t::f16, element::Type_t::i32>(),
        generateParamsForUnsqueezeNegative<element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForUnsqueezeNegative<element::Type_t::i32, element::Type_t::i32>(),
    };

    std::vector<UnsqueezeParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Unsqueeze_With_Hardcoded_Refs,
                         ReferenceUnsqueezeLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForUnsqueeze()),
                         ReferenceUnsqueezeLayerTest::getTestCaseName);

}  // namespace
