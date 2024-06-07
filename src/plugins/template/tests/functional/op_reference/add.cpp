// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct AddParams {
    template <class IT>
    AddParams(const Shape& shape1,
              const Shape& shape2,
              const Shape& outshape,
              const element::Type& iType,
              const std::vector<IT>& iValues1,
              const std::vector<IT>& iValues2,
              const std::vector<IT>& oValues)
        : pshape1(shape1),
          pshape2(shape2),
          inType(iType),
          outType(iType),
          inputData1(CreateTensor(shape1, iType, iValues1)),
          inputData2(CreateTensor(shape2, iType, iValues2)),
          refData(CreateTensor(outshape, iType, oValues)) {}

    Shape pshape1;
    Shape pshape2;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData1;
    ov::Tensor inputData2;
    ov::Tensor refData;
};

class ReferenceAddLayerTest : public testing::TestWithParam<AddParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AddParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape1=" << param.pshape1 << "_";
        result << "shape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape1,
                                                 const Shape& input_shape2,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        const auto add = std::make_shared<op::v1::Add>(in1, in2);
        return std::make_shared<Model>(NodeVector{add}, ParameterVector{in1, in2});
    }
};

class ReferenceAddInPlaceLayerTest : public testing::TestWithParam<AddParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<AddParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape1=" << param.pshape1 << "_";
        result << "shape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape1,
                                                 const Shape& input_shape2,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in1 = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        auto add = std::make_shared<op::v1::Add>(in1, in2);
        add = std::make_shared<op::v1::Add>(add, add);
        add = std::make_shared<op::v1::Add>(add, add);
        add = std::make_shared<op::v1::Add>(add, add);
        return std::make_shared<Model>(NodeVector{add}, ParameterVector{in1, in2});
    }
};

TEST_P(ReferenceAddLayerTest, AddWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceAddInPlaceLayerTest, AddWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<AddParams> generateParamsForAdd() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AddParams> params{
        AddParams(ov::Shape{2, 2},
                  ov::Shape{2, 2},
                  ov::Shape{2, 2},
                  IN_ET,
                  std::vector<T>{1, 2, 3, 4},
                  std::vector<T>{5, 6, 7, 8},
                  std::vector<T>{6, 8, 10, 12}),
        AddParams(ov::Shape{1, 2},
                  ov::Shape{3, 2, 2},
                  ov::Shape{3, 2, 2},
                  IN_ET,
                  std::vector<T>{1, 2},
                  std::vector<T>{5, 6, 7, 8, 2, 3, 1, 5, 6, 7, 1, 3},
                  std::vector<T>{6, 8, 8, 10, 3, 5, 2, 7, 7, 9, 2, 5}),
        AddParams(ov::Shape{1},
                  ov::Shape{1},
                  ov::Shape{1},
                  IN_ET,
                  std::vector<T>{2},
                  std::vector<T>{8},
                  std::vector<T>{10}),
        AddParams(ov::Shape{2, 2},
                  ov::Shape{1},
                  ov::Shape{2, 2},
                  IN_ET,
                  std::vector<T>{2, 4, 7, 8},
                  std::vector<T>{8},
                  std::vector<T>{10, 12, 15, 16}),
    };
    return params;
}

template <element::Type_t IN_ET>
std::vector<AddParams> generateParamsForAddInPlace() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<AddParams> params{AddParams(ov::Shape{2, 2},
                                            ov::Shape{2, 2},
                                            ov::Shape{2, 2},
                                            IN_ET,
                                            std::vector<T>{1, 2, 3, 4},
                                            std::vector<T>{5, 6, 7, 8},
                                            std::vector<T>{48, 64, 80, 96})};
    return params;
}

std::vector<AddParams> generateCombinedParamsForAdd() {
    const std::vector<std::vector<AddParams>> allTypeParams{generateParamsForAdd<element::Type_t::f32>(),
                                                            generateParamsForAdd<element::Type_t::f16>(),
                                                            generateParamsForAdd<element::Type_t::bf16>(),
                                                            generateParamsForAdd<element::Type_t::i64>(),
                                                            generateParamsForAdd<element::Type_t::i32>(),
                                                            generateParamsForAdd<element::Type_t::i16>(),
                                                            generateParamsForAdd<element::Type_t::i8>(),
                                                            generateParamsForAdd<element::Type_t::u64>(),
                                                            generateParamsForAdd<element::Type_t::u32>(),
                                                            generateParamsForAdd<element::Type_t::u16>(),
                                                            generateParamsForAdd<element::Type_t::u8>()};

    std::vector<AddParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<AddParams> generateCombinedParamsForAddInPlace() {
    const std::vector<std::vector<AddParams>> allTypeParams{generateParamsForAddInPlace<element::Type_t::f32>(),
                                                            generateParamsForAddInPlace<element::Type_t::f16>(),
                                                            generateParamsForAddInPlace<element::Type_t::bf16>(),
                                                            generateParamsForAddInPlace<element::Type_t::i64>(),
                                                            generateParamsForAddInPlace<element::Type_t::i32>(),
                                                            generateParamsForAddInPlace<element::Type_t::i16>(),
                                                            generateParamsForAddInPlace<element::Type_t::i8>(),
                                                            generateParamsForAddInPlace<element::Type_t::u64>(),
                                                            generateParamsForAddInPlace<element::Type_t::u32>(),
                                                            generateParamsForAddInPlace<element::Type_t::u16>(),
                                                            generateParamsForAddInPlace<element::Type_t::u8>()};

    std::vector<AddParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Add_With_Hardcoded_Refs,
                         ReferenceAddLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForAdd()),
                         ReferenceAddLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Add_In_Place_With_Hardcoded_Refs,
                         ReferenceAddInPlaceLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForAddInPlace()),
                         ReferenceAddLayerTest::getTestCaseName);

}  // namespace
