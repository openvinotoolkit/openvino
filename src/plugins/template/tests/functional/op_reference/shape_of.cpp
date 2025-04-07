// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shape_of.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct ShapeOfParamsV1 {
    template <class IT, class OT>
    ShapeOfParamsV1(const Shape& input_shape,
                    const Shape& expected_shape,
                    const element::Type& input_type,
                    const element::Type& expected_type,
                    const std::vector<IT>& input_value,
                    const std::vector<OT>& expected_value)
        : m_input_shape(input_shape),
          m_expected_shape(expected_shape),
          m_input_type(input_type),
          m_expected_type(expected_type),
          m_input_value(CreateTensor(input_shape, input_type, input_value)),
          m_expected_value(CreateTensor(expected_shape, expected_type, expected_value)) {}

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
};

struct ShapeOfParamsV3 {
    template <class IT, class OT1, class OT2>
    ShapeOfParamsV3(const Shape& input_shape,
                    const Shape& expected_shape,
                    const element::Type& input_type,
                    const element::Type& expected_type1,
                    const element::Type& expected_type2,
                    const std::vector<IT>& input_value,
                    const std::vector<OT1>& expected_value1,
                    const std::vector<OT2>& expected_value2)
        : m_input_shape(input_shape),
          m_expected_shape(expected_shape),
          m_input_type(input_type),
          m_expected_type1(expected_type1),
          m_expected_type2(expected_type2),
          m_input_value(CreateTensor(input_shape, input_type, input_value)),
          m_expected_value1(CreateTensor(expected_shape, expected_type1, expected_value1)),
          m_expected_value2(CreateTensor(expected_shape, expected_type2, expected_value2)) {}

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type1;
    element::Type m_expected_type2;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value1;
    ov::Tensor m_expected_value2;
};

class ReferenceShapeOfV1LayerTest : public testing::TestWithParam<ShapeOfParamsV1>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params.m_input_type, params.m_input_shape);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ShapeOfParamsV1>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "output_type=" << param.m_expected_type;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const element::Type& input_type, const Shape& input_shape) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto shapeof = std::make_shared<op::v0::ShapeOf>(in);
        return std::make_shared<Model>(NodeVector{shapeof}, ParameterVector{in});
    }
};

class ReferenceShapeOfV3LayerTest : public testing::TestWithParam<ShapeOfParamsV3>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params.m_input_type, params.m_input_shape);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value1, params.m_expected_value2};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ShapeOfParamsV3>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "output_type1=" << param.m_expected_type1 << "; ";
        result << "output_type2=" << param.m_expected_type2 << "; ";

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const element::Type& input_type, const Shape& input_shape) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto shapeof1 = std::make_shared<op::v3::ShapeOf>(in);
        const auto shapeof2 = std::make_shared<op::v3::ShapeOf>(in, element::Type_t::i32);
        return std::make_shared<Model>(OutputVector{shapeof1, shapeof2}, ParameterVector{in});
    }
};

TEST_P(ReferenceShapeOfV1LayerTest, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceShapeOfV3LayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IT, element::Type_t OT>
std::vector<ShapeOfParamsV1> generateParamsForShapeOfSmall_V1() {
    using T1 = typename element_type_traits<IT>::value_type;
    using T2 = typename element_type_traits<OT>::value_type;

    std::vector<ShapeOfParamsV1> params{
        ShapeOfParamsV1(Shape{2}, Shape{1}, IT, OT, std::vector<T1>{2, 0}, std::vector<T2>{2}),
        ShapeOfParamsV1(Shape{2, 4}, Shape{2}, IT, OT, std::vector<T1>{2 * 4, 0}, std::vector<T2>{2, 4})};

    return params;
}

template <element::Type_t IT, element::Type_t OT>
std::vector<ShapeOfParamsV1> generateParamsForShapeOfBig_V1() {
    using T1 = typename element_type_traits<IT>::value_type;
    using T2 = typename element_type_traits<OT>::value_type;

    std::vector<ShapeOfParamsV1> params{
        ShapeOfParamsV1(Shape{2}, Shape{1}, IT, OT, std::vector<T1>{2, 0}, std::vector<T2>{2}),
        ShapeOfParamsV1(Shape{2, 4}, Shape{2}, IT, OT, std::vector<T1>{2 * 4, 0}, std::vector<T2>{2, 4}),
        ShapeOfParamsV1(Shape{2, 4, 8, 16, 32},
                        Shape{5},
                        IT,
                        OT,
                        std::vector<T1>{2 * 4 * 8 * 16 * 32, 0},
                        std::vector<T2>{2, 4, 8, 16, 32})};

    return params;
}

template <element::Type_t IT, element::Type_t OT1, element::Type_t OT2>
std::vector<ShapeOfParamsV3> generateParamsForShapeOfSmall_V3() {
    using T1 = typename element_type_traits<IT>::value_type;
    using T2 = typename element_type_traits<OT1>::value_type;
    using T3 = typename element_type_traits<OT2>::value_type;

    std::vector<ShapeOfParamsV3> params{ShapeOfParamsV3(Shape{2},
                                                        Shape{1},
                                                        IT,
                                                        OT1,
                                                        OT2,
                                                        std::vector<T1>{2, 0},
                                                        std::vector<T2>{2},
                                                        std::vector<T3>{2}),
                                        ShapeOfParamsV3(Shape{2, 4},
                                                        Shape{2},
                                                        IT,
                                                        OT1,
                                                        OT2,
                                                        std::vector<T1>{2 * 4, 0},
                                                        std::vector<T2>{2, 4},
                                                        std::vector<T3>{2, 4})};

    return params;
}

template <element::Type_t IT, element::Type_t OT1, element::Type_t OT2>
std::vector<ShapeOfParamsV3> generateParamsForShapeOfBig_V3() {
    using T1 = typename element_type_traits<IT>::value_type;
    using T2 = typename element_type_traits<OT1>::value_type;
    using T3 = typename element_type_traits<OT2>::value_type;

    std::vector<ShapeOfParamsV3> params{ShapeOfParamsV3(Shape{2},
                                                        Shape{1},
                                                        IT,
                                                        OT1,
                                                        OT2,
                                                        std::vector<T1>{2, 0},
                                                        std::vector<T2>{2},
                                                        std::vector<T3>{2}),
                                        ShapeOfParamsV3(Shape{2, 4},
                                                        Shape{2},
                                                        IT,
                                                        OT1,
                                                        OT2,
                                                        std::vector<T1>{2 * 4, 0},
                                                        std::vector<T2>{2, 4},
                                                        std::vector<T3>{2, 4}),
                                        ShapeOfParamsV3(Shape{2, 4, 8, 16, 32},
                                                        Shape{5},
                                                        IT,
                                                        OT1,
                                                        OT2,
                                                        std::vector<T1>{2 * 4 * 8 * 16 * 32, 0},
                                                        std::vector<T2>{2, 4, 8, 16, 32},
                                                        std::vector<T3>{2, 4, 8, 16, 32})};

    return params;
}

std::vector<ShapeOfParamsV1> generateCombinedParamsForShapeOfV1() {
    const std::vector<std::vector<ShapeOfParamsV1>> allTypeParams{
        generateParamsForShapeOfBig_V1<element::Type_t::f32, element::Type_t::i64>(),
        generateParamsForShapeOfBig_V1<element::Type_t::f16, element::Type_t::i64>(),
        generateParamsForShapeOfBig_V1<element::Type_t::bf16, element::Type_t::i64>(),
        generateParamsForShapeOfBig_V1<element::Type_t::i64, element::Type_t::i64>(),
        generateParamsForShapeOfBig_V1<element::Type_t::i32, element::Type_t::i64>(),
        generateParamsForShapeOfBig_V1<element::Type_t::u64, element::Type_t::i64>(),
        generateParamsForShapeOfBig_V1<element::Type_t::u32, element::Type_t::i64>(),
        generateParamsForShapeOfSmall_V1<element::Type_t::i16, element::Type_t::i64>(),
        generateParamsForShapeOfSmall_V1<element::Type_t::i8, element::Type_t::i64>(),
        generateParamsForShapeOfSmall_V1<element::Type_t::u16, element::Type_t::i64>(),
        generateParamsForShapeOfSmall_V1<element::Type_t::u8, element::Type_t::i64>()};

    std::vector<ShapeOfParamsV1> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<ShapeOfParamsV3> generateCombinedParamsForShapeOfV3() {
    const std::vector<std::vector<ShapeOfParamsV3>> allTypeParams{
        generateParamsForShapeOfBig_V3<element::Type_t::f32, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfBig_V3<element::Type_t::f16, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfBig_V3<element::Type_t::bf16, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfBig_V3<element::Type_t::i64, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfBig_V3<element::Type_t::i32, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfBig_V3<element::Type_t::u64, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfBig_V3<element::Type_t::u32, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfSmall_V3<element::Type_t::i16, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfSmall_V3<element::Type_t::i8, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfSmall_V3<element::Type_t::u16, element::Type_t::i64, element::Type_t::i32>(),
        generateParamsForShapeOfSmall_V3<element::Type_t::u8, element::Type_t::i64, element::Type_t::i32>()};

    std::vector<ShapeOfParamsV3> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_With_Hardcoded_Refs,
                         ReferenceShapeOfV1LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForShapeOfV1()),
                         ReferenceShapeOfV1LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_With_Hardcoded_Refs,
                         ReferenceShapeOfV3LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForShapeOfV3()),
                         ReferenceShapeOfV3LayerTest::getTestCaseName);

}  // namespace
