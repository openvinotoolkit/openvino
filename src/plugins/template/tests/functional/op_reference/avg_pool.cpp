// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/avg_pool.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct AvgPoolParams {
    template <class IT>
    AvgPoolParams(const Shape& input_shape,
                  const Shape& output_shape,
                  const element::Type& input_type,
                  const element::Type& ouput_type,
                  const std::vector<IT>& input_values,
                  const std::vector<IT>& output_values,
                  const Strides& strides,
                  const Shape& pads_begin,
                  const Shape& pads_end,
                  const Shape& kernel,
                  const bool exclude_pad,
                  const op::RoundingType& rounding_type,
                  const op::PadType& pad_type)
        : m_input_shape(input_shape),
          m_output_shape(output_shape),
          m_input_type(input_type),
          m_output_type(ouput_type),
          m_input_data(CreateTensor(input_shape, input_type, input_values)),
          m_expected_data(CreateTensor(output_shape, ouput_type, output_values)),
          m_strides(strides),
          m_pads_begin(pads_begin),
          m_pads_end(pads_end),
          m_kernel(kernel),
          m_exclude_pad(exclude_pad),
          m_rounding_type(rounding_type),
          m_pad_type(pad_type) {}

    Shape m_input_shape;
    Shape m_output_shape;
    element::Type m_input_type;
    element::Type m_output_type;
    ov::Tensor m_input_data;
    ov::Tensor m_expected_data;
    Strides m_strides;
    Shape m_pads_begin;
    Shape m_pads_end;
    Shape m_kernel;
    bool m_exclude_pad;
    op::RoundingType m_rounding_type;
    op::PadType m_pad_type;
};

class ReferenceAvgPoolLayerTestV1 : public testing::TestWithParam<AvgPoolParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.m_input_shape,
                                  params.m_input_type,
                                  params.m_strides,
                                  params.m_pads_begin,
                                  params.m_pads_end,
                                  params.m_kernel,
                                  params.m_exclude_pad,
                                  params.m_rounding_type,
                                  params.m_pad_type);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AvgPoolParams>& obj) {
        const auto& params = obj.param;
        std::ostringstream result;
        result << "iShape=" << params.m_input_shape << "_";
        result << "iType=" << params.m_input_type << "_";
        result << "oShape=" << params.m_output_shape << "_";
        result << "oType=" << params.m_output_type << "_";
        result << "excludePad=" << params.m_exclude_pad << "_";
        result << "roundingType=" << params.m_rounding_type << "_";
        result << "padType=" << params.m_pad_type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const Strides& strides,
                                                 const Shape& pads_begin,
                                                 const Shape& pads_end,
                                                 const Shape& kernel,
                                                 const bool exclude_pad,
                                                 const op::RoundingType rounding_type,
                                                 const op::PadType pad_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto avgPool = std::make_shared<op::v1::AvgPool>(in,
                                                               strides,
                                                               pads_begin,
                                                               pads_end,
                                                               kernel,
                                                               exclude_pad,
                                                               rounding_type,
                                                               pad_type);
        return std::make_shared<Model>(NodeVector{avgPool}, ParameterVector{in});
    }
};

TEST_P(ReferenceAvgPoolLayerTestV1, AvgPoolWithHardcodedRefs) {
    Exec();
}

template <typename T>
std::vector<T> getContinuousIncreasingValue(size_t elementSize, float step) {
    std::vector<T> a(elementSize);
    std::iota(std::begin(a), std::end(a), step);
    return a;
}

template <element::Type_t IN_ET>
std::vector<AvgPoolParams> generateParamsForAvgPoolV1() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AvgPoolParams> params{
        AvgPoolParams(ov::Shape{1, 1, 5},
                      ov::Shape{1, 1, 5},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5},
                      std::vector<T>{1.5, 2.5, 3.5, 4.5, 5},
                      Strides{1},
                      Shape{0},
                      Shape{1},
                      Shape{2},
                      true,
                      op::RoundingType::FLOOR,
                      op::PadType::EXPLICIT),
        AvgPoolParams(ov::Shape{1, 1, 8},
                      ov::Shape{1, 1, 4},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                      std::vector<T>{2, 4, 6, 7.5},
                      Strides{2},
                      Shape{0},
                      Shape{0},
                      Shape{3},
                      false,
                      op::RoundingType::CEIL,
                      op::PadType::EXPLICIT),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 2, 2},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<T>{3, 4, 6, 7},
                      Strides{1, 1},
                      Shape{0, 0},
                      Shape{0, 0},
                      Shape{2, 2},
                      true,
                      op::RoundingType::FLOOR,
                      op::PadType::NOTSET),
        AvgPoolParams(ov::Shape{1, 1, 4, 4},
                      ov::Shape{1, 1, 2, 2},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                      std::vector<T>{6, 7, 10, 11},
                      Strides{1, 1},
                      Shape{0, 0},
                      Shape{0, 0},
                      Shape{3, 3},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::NOTSET),
        AvgPoolParams(ov::Shape{1, 1, 2, 2},
                      ov::Shape{1, 1, 3, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4},
                      std::vector<T>{1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4},
                      Strides{1, 1},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{2, 2},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::NOTSET),
        AvgPoolParams(ov::Shape{1, 1, 1, 5},
                      ov::Shape{1, 1, 1, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5},
                      std::vector<T>{1.5, 3, 4.5},
                      Strides{1, 2},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{3, 3},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::EXPLICIT),
        AvgPoolParams(ov::Shape{1, 1, 1, 5},
                      ov::Shape{1, 1, 1, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{2.5, 2, 12, 4, 5},
                      std::vector<T>{0.5, 2, 1},
                      Strides{1, 2},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{3, 3},
                      false,
                      op::RoundingType::CEIL,
                      op::PadType::EXPLICIT),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 3, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<T>{3, 4, 2.25, 6, 7, 3.75, 3.75, 4.25, 2.25},
                      Strides{1, 1},
                      Shape{0, 0},
                      Shape{0, 0},
                      Shape{2, 2},
                      false,
                      op::RoundingType::CEIL,
                      op::PadType::SAME_UPPER),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 3, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<T>{0.25, 0.75, 1.25, 1.25, 3, 4, 2.75, 6, 7},
                      Strides{1, 1},
                      Shape{0, 0},
                      Shape{0, 0},
                      Shape{2, 2},
                      false,
                      op::RoundingType::CEIL,
                      op::PadType::SAME_LOWER),
        AvgPoolParams(ov::Shape{1, 1, 2, 2, 2},
                      ov::Shape{1, 1, 2, 2, 1},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                      std::vector<T>{1.5, 3.5, 5.5, 7.5},
                      Strides{1, 1, 1},
                      Shape{0, 0, 0},
                      Shape{0, 0, 0},
                      Shape{1, 1, 2},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::VALID),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 3, 3},
                      IN_ET,
                      IN_ET,
                      getContinuousIncreasingValue<T>(1 * 1 * 3 * 3, 1),
                      std::vector<T>{1.0f, 2.5f, 0, 5.5f, 7.0f, 0, 0, 0, 0},
                      Strides{2, 2},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{2, 2},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::NOTSET),
    };
    return params;
}

std::vector<AvgPoolParams> generateCombinedParamsForAvgPoolV1() {
    const std::vector<std::vector<AvgPoolParams>> allTypeParams{generateParamsForAvgPoolV1<element::Type_t::f32>(),
                                                                generateParamsForAvgPoolV1<element::Type_t::f16>(),
                                                                generateParamsForAvgPoolV1<element::Type_t::bf16>()};

    std::vector<AvgPoolParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_With_Hardcoded_Refs,
                         ReferenceAvgPoolLayerTestV1,
                         ::testing::ValuesIn(generateCombinedParamsForAvgPoolV1()),
                         ReferenceAvgPoolLayerTestV1::getTestCaseName);

class ReferenceAvgPoolLayerTestV14 : public testing::TestWithParam<AvgPoolParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.m_input_shape,
                                  params.m_input_type,
                                  params.m_strides,
                                  params.m_pads_begin,
                                  params.m_pads_end,
                                  params.m_kernel,
                                  params.m_exclude_pad,
                                  params.m_rounding_type,
                                  params.m_pad_type);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AvgPoolParams>& obj) {
        const auto& params = obj.param;
        std::ostringstream result;
        result << "iShape=" << params.m_input_shape << "_";
        result << "iType=" << params.m_input_type << "_";
        result << "oShape=" << params.m_output_shape << "_";
        result << "oType=" << params.m_output_type << "_";
        result << "excludePad=" << params.m_exclude_pad << "_";
        result << "roundingType=" << params.m_rounding_type << "_";
        result << "padType=" << params.m_pad_type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const Strides& strides,
                                                 const Shape& pads_begin,
                                                 const Shape& pads_end,
                                                 const Shape& kernel,
                                                 const bool exclude_pad,
                                                 const op::RoundingType rounding_type,
                                                 const op::PadType pad_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto avgPool = std::make_shared<op::v14::AvgPool>(in,
                                                                strides,
                                                                pads_begin,
                                                                pads_end,
                                                                kernel,
                                                                exclude_pad,
                                                                rounding_type,
                                                                pad_type);
        return std::make_shared<Model>(NodeVector{avgPool}, ParameterVector{in});
    }
};

TEST_P(ReferenceAvgPoolLayerTestV14, AvgPoolWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<AvgPoolParams> generateParamsForAvgPoolV14() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AvgPoolParams> params{
        AvgPoolParams(ov::Shape{1, 1, 5},
                      ov::Shape{1, 1, 5},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5},
                      std::vector<T>{1.5, 2.5, 3.5, 4.5, 5},
                      Strides{1},
                      Shape{0},
                      Shape{1},
                      Shape{2},
                      true,
                      op::RoundingType::FLOOR,
                      op::PadType::EXPLICIT),
        AvgPoolParams(ov::Shape{1, 1, 8},
                      ov::Shape{1, 1, 4},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                      std::vector<T>{2, 4, 6, 7.5},
                      Strides{2},
                      Shape{0},
                      Shape{0},
                      Shape{3},
                      false,
                      op::RoundingType::CEIL,
                      op::PadType::EXPLICIT),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 2, 2},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<T>{3, 4, 6, 7},
                      Strides{1, 1},
                      Shape{0, 0},
                      Shape{0, 0},
                      Shape{2, 2},
                      true,
                      op::RoundingType::FLOOR,
                      op::PadType::NOTSET),
        AvgPoolParams(ov::Shape{1, 1, 4, 4},
                      ov::Shape{1, 1, 2, 2},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                      std::vector<T>{6, 7, 10, 11},
                      Strides{1, 1},
                      Shape{0, 0},
                      Shape{0, 0},
                      Shape{3, 3},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::NOTSET),
        AvgPoolParams(ov::Shape{1, 1, 2, 2},
                      ov::Shape{1, 1, 3, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4},
                      std::vector<T>{1, 1.5, 2, 2, 2.5, 3, 3, 3.5, 4},
                      Strides{1, 1},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{2, 2},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::NOTSET),
        AvgPoolParams(ov::Shape{1, 1, 1, 5},
                      ov::Shape{1, 1, 1, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5},
                      std::vector<T>{1.5, 3, 4.5},
                      Strides{1, 2},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{3, 3},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::EXPLICIT),
        AvgPoolParams(ov::Shape{1, 1, 1, 5},
                      ov::Shape{1, 1, 1, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{2.5, 2, 12, 4, 5},
                      std::vector<T>{0.5, 2, 1},
                      Strides{1, 2},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{3, 3},
                      false,
                      op::RoundingType::CEIL,
                      op::PadType::EXPLICIT),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 3, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<T>{3, 4, 2.25, 6, 7, 3.75, 3.75, 4.25, 2.25},
                      Strides{1, 1},
                      Shape{0, 0},
                      Shape{0, 0},
                      Shape{2, 2},
                      false,
                      op::RoundingType::CEIL,
                      op::PadType::SAME_UPPER),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 3, 3},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9},
                      std::vector<T>{0.25, 0.75, 1.25, 1.25, 3, 4, 2.75, 6, 7},
                      Strides{1, 1},
                      Shape{0, 0},
                      Shape{0, 0},
                      Shape{2, 2},
                      false,
                      op::RoundingType::CEIL,
                      op::PadType::SAME_LOWER),
        AvgPoolParams(ov::Shape{1, 1, 2, 2, 2},
                      ov::Shape{1, 1, 2, 2, 1},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                      std::vector<T>{1.5, 3.5, 5.5, 7.5},
                      Strides{1, 1, 1},
                      Shape{0, 0, 0},
                      Shape{0, 0, 0},
                      Shape{1, 1, 2},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::VALID),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 3, 3},
                      IN_ET,
                      IN_ET,
                      getContinuousIncreasingValue<T>(1 * 1 * 3 * 3, 1),
                      std::vector<T>{1.0f, 2.5f, 0, 5.5f, 7.0f, 0, 0, 0, 0},
                      Strides{2, 2},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{2, 2},
                      true,
                      op::RoundingType::CEIL,
                      op::PadType::NOTSET),
        AvgPoolParams(ov::Shape{1, 1, 2, 2, 2},
                      ov::Shape{1, 1, 2, 2, 1},
                      IN_ET,
                      IN_ET,
                      std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                      std::vector<T>{1.5, 3.5, 5.5, 7.5},
                      Strides{1, 1, 1},
                      Shape{0, 0, 0},
                      Shape{0, 0, 0},
                      Shape{1, 1, 2},
                      true,
                      op::RoundingType::CEIL_TORCH,
                      op::PadType::VALID),
        AvgPoolParams(ov::Shape{1, 1, 3, 3},
                      ov::Shape{1, 1, 2, 2},
                      IN_ET,
                      IN_ET,
                      getContinuousIncreasingValue<T>(1 * 1 * 3 * 3, 1),
                      std::vector<T>{1.0f, 2.5f, 5.5f, 7.0f},
                      Strides{2, 2},
                      Shape{1, 1},
                      Shape{1, 1},
                      Shape{2, 2},
                      true,
                      op::RoundingType::CEIL_TORCH,
                      op::PadType::EXPLICIT),
    };
    return params;
}

std::vector<AvgPoolParams> generateCombinedParamsForAvgPoolV14() {
    const std::vector<std::vector<AvgPoolParams>> allTypeParams{generateParamsForAvgPoolV14<element::Type_t::f32>(),
                                                                generateParamsForAvgPoolV14<element::Type_t::f16>(),
                                                                generateParamsForAvgPoolV14<element::Type_t::bf16>()};

    std::vector<AvgPoolParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_With_Hardcoded_Refs,
                         ReferenceAvgPoolLayerTestV14,
                         ::testing::ValuesIn(generateCombinedParamsForAvgPoolV14()),
                         ReferenceAvgPoolLayerTestV14::getTestCaseName);
}  // namespace
