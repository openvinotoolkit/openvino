// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct SqueezeParams {
    template <class IO_T, class Axes_T>
    SqueezeParams(const Shape& input_shape,
                  const Shape& output_shape,
                  const element::Type& input_type,
                  const element::Type& output_type,
                  const std::vector<IO_T>& input_value,
                  const std::vector<IO_T>& expected_value,
                  const Shape& axes_shape,
                  const element::Type& axes_type,
                  const std::vector<Axes_T>& axes_value)
        : m_input_shape(input_shape),
          m_output_shape(output_shape),
          m_input_type(input_type),
          m_output_type(output_type),
          m_input_value(CreateTensor(input_shape, input_type, input_value)),
          m_expected_value(CreateTensor(output_shape, output_type, expected_value)),
          m_axes_shape(axes_shape),
          m_axes_type(axes_type),
          m_axes_value(CreateTensor(axes_type, axes_value)),
          m_axes_node(true) {}

    template <class IO_T>
    SqueezeParams(const Shape& input_shape,
                  const Shape& output_shape,
                  const element::Type& input_type,
                  const element::Type& output_type,
                  const std::vector<IO_T>& input_value,
                  const std::vector<IO_T>& expected_value)
        : m_input_shape(input_shape),
          m_output_shape(output_shape),
          m_input_type(input_type),
          m_output_type(output_type),
          m_input_value(CreateTensor(input_shape, input_type, input_value)),
          m_expected_value(CreateTensor(output_shape, input_type, expected_value)),
          m_axes_node(false) {}

    Shape m_input_shape;
    Shape m_output_shape;
    element::Type m_input_type;
    element::Type m_output_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
    Shape m_axes_shape;
    element::Type m_axes_type;
    ov::Tensor m_axes_value;
    bool m_axes_node;
};

class ReferenceSqueezeLayerTest : public testing::TestWithParam<SqueezeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SqueezeParams>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape << "; ";
        result << "output_shape=" << param.m_output_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "output_type=" << param.m_output_type << "; ";
        if (param.m_axes_node) {
            result << "axes_shape=" << param.m_axes_shape << "; ";
            result << "axes_type=" << param.m_axes_type;
        }
        result << "axes_node=" << param.m_axes_node;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const SqueezeParams& params) {
        const auto in = std::make_shared<op::v0::Parameter>(params.m_input_type, params.m_input_shape);
        std::shared_ptr<op::v0::Constant> axes_node = NULL;
        std::shared_ptr<op::v0::Squeeze> squeeze = NULL;
        if (params.m_axes_node) {
            axes_node =
                std::make_shared<op::v0::Constant>(params.m_axes_type, params.m_axes_shape, params.m_axes_value.data());
            squeeze = std::make_shared<op::v0::Squeeze>(in, axes_node);
        } else {
            squeeze = std::make_shared<op::v0::Squeeze>(in);
        }

        return std::make_shared<ov::Model>(squeeze, ParameterVector{in});
    }
};

TEST_P(ReferenceSqueezeLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IO_T, element::Type_t Axes_T>
std::vector<SqueezeParams> generateParamsForSqueeze() {
    using T1 = typename element_type_traits<IO_T>::value_type;
    using T2 = typename element_type_traits<Axes_T>::value_type;

    std::vector<SqueezeParams> params{
        SqueezeParams(Shape{1, 4, 1, 1, 2},
                      Shape{4, 1, 2},
                      IO_T,
                      IO_T,
                      std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                      std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                      Shape{2},
                      Axes_T,
                      std::vector<T2>{0, 2}),
        SqueezeParams(Shape{1, 4, 1, 1, 2},
                      Shape{4, 2},
                      IO_T,
                      IO_T,
                      std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                      std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                      Shape{0},
                      Axes_T,
                      std::vector<T2>{}),
        SqueezeParams(Shape{1, 4, 1, 1, 2},
                      Shape{4, 2},
                      IO_T,
                      IO_T,
                      std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8},
                      std::vector<T1>{1, 2, 3, 4, 5, 6, 7, 8}),
    };

    return params;
}

std::vector<SqueezeParams> generateCombinedParamsForSqueeze() {
    const std::vector<std::vector<SqueezeParams>> allTypeParams{
        generateParamsForSqueeze<element::Type_t::f32, element::Type_t::i64>(),
        generateParamsForSqueeze<element::Type_t::i64, element::Type_t::i64>(),
        generateParamsForSqueeze<element::Type_t::i32, element::Type_t::i64>(),
        generateParamsForSqueeze<element::Type_t::i16, element::Type_t::i64>(),
        generateParamsForSqueeze<element::Type_t::i8, element::Type_t::i64>(),
        generateParamsForSqueeze<element::Type_t::u64, element::Type_t::i64>(),
        generateParamsForSqueeze<element::Type_t::u32, element::Type_t::i64>(),
        generateParamsForSqueeze<element::Type_t::u16, element::Type_t::i64>(),
        generateParamsForSqueeze<element::Type_t::u8, element::Type_t::i64>()};

    std::vector<SqueezeParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Squeeze_With_Hardcoded_Refs,
                         ReferenceSqueezeLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForSqueeze()),
                         ReferenceSqueezeLayerTest::getTestCaseName);

}  // namespace
