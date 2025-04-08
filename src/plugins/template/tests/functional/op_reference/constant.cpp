// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"

#include <gtest/gtest.h>

#include <random>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/equal.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ConstantParams {
    template <class IT, class OT>
    ConstantParams(const Shape& inputShape,
                   const element::Type& inType,
                   const element::Type& refType,
                   const std::vector<IT>& inputData,
                   const std::vector<OT>& refData,
                   const std::string& test_name = "")
        : inputShape(inputShape),
          inType(inType),
          refType(refType),
          inputData(CreateTensor(inputShape, inType, inputData)),
          refData(CreateTensor(inputShape, refType, refData)),
          testcaseName(test_name) {}

    Shape inputShape;
    element::Type inType;
    element::Type refType;
    ov::Tensor inputData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferenceConstantLayerTest : public testing::TestWithParam<ConstantParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConstantParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iShape=" << param.inputShape << "_";
        result << "iType=" << param.inType << "_";
        if (param.testcaseName != "") {
            result << "oType=" << param.refType << "_";
            result << param.testcaseName;
        } else {
            result << "oType=" << param.refType;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ParamType& params) {
        auto A = op::v0::Constant::create(params.inType, params.inputShape, params.inputData.data());
        return std::make_shared<Model>(A, ParameterVector{});
    }
};

class ReferenceConstantLayerTest_2Constant : public ReferenceConstantLayerTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        refOutData = {params.refData, params.refData};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ParamType& params) {
        auto A = op::v0::Constant::create(params.inType, params.inputShape, params.inputData.data());
        auto B = op::v0::Constant::create(params.inType, params.inputShape, params.inputData.data());
        return std::make_shared<Model>(NodeVector{A, B}, ParameterVector{});
    }
};

class ReferenceConstantLayerTest_WithOp : public ReferenceConstantLayerTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        refOutData = {params.refData};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ParamType& params) {
        auto A = op::v0::Constant::create(params.inType, params.inputShape, params.inputData.data());
        return std::make_shared<Model>(std::make_shared<op::v0::Abs>(A), ParameterVector{});
    }
};

class ReferenceConstantLayerTest_MultiUse : public ReferenceConstantLayerTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        refOutData = {params.refData};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ParamType& params) {
        const auto A = std::make_shared<op::v0::Constant>(
            params.inType,
            params.inputShape,
            std::vector<std::string>{std::to_string(*reinterpret_cast<int*>(params.inputData.data()))});
        return std::make_shared<Model>(A, ParameterVector{});
    }
};

class ReferenceConstantLayerTest_EqualityBool : public ReferenceConstantLayerTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        refOutData = {params.refData};
    }

protected:
    static std::shared_ptr<Model> CreateFunction(const ParamType& params) {
        auto A = op::v0::Constant::create(params.inType, params.inputShape, params.inputData.data());
        auto B = op::v0::Constant::create(params.inType, params.inputShape, {true, true, true, true});
        return std::make_shared<Model>(std::make_shared<op::v1::Equal>(A, B), ParameterVector{});
    }
};

TEST_P(ReferenceConstantLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceConstantLayerTest_2Constant, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceConstantLayerTest_WithOp, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceConstantLayerTest_MultiUse, CompareWithHardcodedRefs) {
    Exec();
    Infer();
    Validate();
}

TEST_P(ReferenceConstantLayerTest_EqualityBool, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ConstantParams> generateConstantParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ConstantParams> constantParams{
        // tensor_constant
        ConstantParams({2, 2, 2},
                       IN_ET,
                       IN_ET,
                       std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                       std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                       "tensor_constant"),
    };
    return constantParams;
}

std::vector<ConstantParams> generateConstantDefinedTypeParams() {
    std::vector<ConstantParams> constantParams{
        // scalar_constant_float32
        ConstantParams({},
                       element::Type_t::f32,
                       element::Type_t::f32,
                       std::vector<float>{4.75},
                       std::vector<float>{4.75f},
                       "scalar_constant_float32"),
        // scalar_constant_int64
        ConstantParams({},
                       element::Type_t::i64,
                       element::Type_t::i64,
                       std::vector<int64_t>{0x4000000000000001},
                       std::vector<int64_t>{0x4000000000000001},
                       "scalar_constant_int64"),
        // tensor_constant_float32
        ConstantParams({2, 2},
                       element::Type_t::f32,
                       element::Type_t::f32,
                       std::vector<float>{4.75, 4.5, -5.25, 0.0},
                       std::vector<float>{4.75f, 4.5f, -5.25f, 0.0f},
                       "tensor_constant_float32"),
        // tensor_constant_int64
        ConstantParams({2},
                       element::Type_t::i64,
                       element::Type_t::i64,
                       std::vector<int64_t>{0x4000000000000001, 0x4000000000000002},
                       std::vector<int64_t>{0x4000000000000001, 0x4000000000000002},
                       "tensor_constant_int64"),
        ConstantParams(
            {3, 9},
            element::Type_t::f8e4m3,
            element::Type_t::f8e4m3,
            std::vector<ov::float8_e4m3>{4.75f, 4.5f,  -5.25f, 0.0f,  0.1f,  0.2f,  0.3f,  0.4f,         0.5f,
                                         0.6f,  0.7f,  0.8f,   0.9f,  1.f,   -0.0f, -0.1f, -0.2f,        -0.3f,
                                         -0.4f, -0.5f, -0.6f,  -0.7f, -0.8f, -0.9f, -1.f,  0.001953125f, 448.f},
            std::vector<ov::float8_e4m3>{5.0f,     4.5f,        -5.0f,      0.0f,     0.1015625f,   0.203125f, 0.3125f,
                                         0.40625f, 0.5f,        0.625f,     0.6875f,  0.8125f,      0.875f,    1.f,
                                         -0.f,     -0.1015625f, -0.203125f, -0.3125f, -0.40625f,    -0.5f,     -0.625f,
                                         -0.6875f, -0.8125f,    -0.875f,    -1.f,     0.001953125f, 448.f},
            "tensor_constant_f8e4m3"),
        ConstantParams({3, 9},
                       element::Type_t::f8e5m2,
                       element::Type_t::f8e5m2,
                       std::vector<ov::float8_e5m2>{4.75f,  4.5f,
                                                    -5.25f, 0.0f,
                                                    0.1f,   0.2f,
                                                    0.3f,   0.4f,
                                                    0.5f,   0.6f,
                                                    0.7f,   0.8f,
                                                    0.9f,   1.f,
                                                    -0.0f,  -0.1f,
                                                    -0.2f,  -0.3f,
                                                    -0.4f,  -0.5f,
                                                    -0.6f,  -0.7f,
                                                    -0.8f,  -0.9f,
                                                    -1.f,   0.0000152587890625f,
                                                    57344.f},
                       std::vector<ov::float8_e5m2>{4.75f,    4.5f,
                                                    -5.25f,   0.0f,
                                                    0.09375f, 0.1875f,
                                                    0.3125f,  0.375f,
                                                    0.5f,     0.625f,
                                                    0.75f,    0.75f,
                                                    0.875f,   1.f,
                                                    -0.f,     -0.09375f,
                                                    -0.1875f, -0.3125f,
                                                    -0.375f,  -0.5f,
                                                    -0.625f,  -0.75f,
                                                    -0.75f,   -0.875f,
                                                    -1.f,     0.0000152587890625f,
                                                    57344.f},
                       "tensor_constant_f8e5m2"),
    };
    return constantParams;
}

std::vector<ConstantParams> generateConstantCombinedParams() {
    const std::vector<std::vector<ConstantParams>> constantTypeParams{
        generateConstantParams<element::Type_t::i8>(),
        generateConstantParams<element::Type_t::i16>(),
        generateConstantParams<element::Type_t::i32>(),
        generateConstantParams<element::Type_t::i64>(),
        generateConstantParams<element::Type_t::u8>(),
        generateConstantParams<element::Type_t::u16>(),
        generateConstantParams<element::Type_t::u32>(),
        generateConstantParams<element::Type_t::u64>(),
        generateConstantParams<element::Type_t::bf16>(),
        generateConstantParams<element::Type_t::f16>(),
        generateConstantParams<element::Type_t::f32>(),
        generateConstantParams<element::Type_t::f64>(),
        generateConstantDefinedTypeParams(),
    };
    std::vector<ConstantParams> combinedParams;

    for (const auto& params : constantTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<ConstantParams> generateConstant2ConstantCombinedParams() {
    const std::vector<std::vector<ConstantParams>> constantTypeParams{
        generateConstantParams<element::Type_t::i8>(),
        generateConstantParams<element::Type_t::i16>(),
        generateConstantParams<element::Type_t::i32>(),
        generateConstantParams<element::Type_t::i64>(),
        generateConstantParams<element::Type_t::u8>(),
        generateConstantParams<element::Type_t::u16>(),
        generateConstantParams<element::Type_t::u32>(),
        generateConstantParams<element::Type_t::u64>(),
        generateConstantParams<element::Type_t::bf16>(),
        generateConstantParams<element::Type_t::f16>(),
        generateConstantParams<element::Type_t::f32>(),
        generateConstantParams<element::Type_t::f64>(),
    };
    std::vector<ConstantParams> combinedParams;

    for (const auto& params : constantTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t IN_ET>
std::vector<ConstantParams> generateConstantWithOpParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ConstantParams> constantParams{
        // tensor_constant_with_op
        ConstantParams({2, 2, 2},
                       IN_ET,
                       IN_ET,
                       std::vector<T>{-1, 2, 3, -4, 5, -6, -7, 8},
                       std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                       "tensor_constant_with_op"),
    };
    return constantParams;
}

std::vector<ConstantParams> generateConstantWithOpCombinedParams() {
    const std::vector<std::vector<ConstantParams>> constantTypeParams{
        generateConstantWithOpParams<element::Type_t::i8>(),
        generateConstantWithOpParams<element::Type_t::i16>(),
        generateConstantWithOpParams<element::Type_t::i32>(),
        generateConstantWithOpParams<element::Type_t::i64>(),
        generateConstantWithOpParams<element::Type_t::bf16>(),
        generateConstantWithOpParams<element::Type_t::f16>(),
        generateConstantWithOpParams<element::Type_t::f32>(),
        generateConstantWithOpParams<element::Type_t::f64>(),
    };
    std::vector<ConstantParams> combinedParams;

    for (const auto& params : constantTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<ConstantParams> generateConstantDefinedTypeMultiUseCombinedParams() {
    const std::vector<ConstantParams> combinedParams{
        // constant_multi_use
        ConstantParams({},
                       element::Type_t::i32,
                       element::Type_t::i32,
                       std::vector<int32_t>{388},
                       std::vector<int32_t>{388},
                       "constant_multi_use"),
    };
    return combinedParams;
}

std::vector<ConstantParams> generateConstantDefinedTypeEqualityBoolCombinedParams() {
    const std::vector<ConstantParams> combinedParams{
        // constant_equality_bool
        ConstantParams({4},
                       element::Type_t::boolean,
                       element::Type_t::boolean,
                       std::vector<char>{true, false, true, false},
                       std::vector<char>{true, false, true, false},
                       "constant_equality_bool"),
    };
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs,
                         ReferenceConstantLayerTest,
                         testing::ValuesIn(generateConstantCombinedParams()),
                         ReferenceConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs,
                         ReferenceConstantLayerTest_2Constant,
                         testing::ValuesIn(generateConstant2ConstantCombinedParams()),
                         ReferenceConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs,
                         ReferenceConstantLayerTest_WithOp,
                         testing::ValuesIn(generateConstantWithOpCombinedParams()),
                         ReferenceConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs,
                         ReferenceConstantLayerTest_MultiUse,
                         testing::ValuesIn(generateConstantDefinedTypeMultiUseCombinedParams()),
                         ReferenceConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs,
                         ReferenceConstantLayerTest_EqualityBool,
                         testing::ValuesIn(generateConstantDefinedTypeEqualityBoolCombinedParams()),
                         ReferenceConstantLayerTest::getTestCaseName);
}  // namespace
