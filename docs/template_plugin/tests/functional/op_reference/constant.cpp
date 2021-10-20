// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>
#include "openvino/op/constant.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/equal.hpp"
#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ConstantParams {
    template <class IT, class OT>
    ConstantParams(const PartialShape& inputShape,
                   const element::Type& inType, const element::Type& refType,
                   const std::vector<IT>& inputData, const std::vector<OT>& refData,
                   const std::string& test_name = "")
        : inputShape(inputShape),
          inType(inType),
          refType(refType),
          inputData(CreateTensor(inType, inputData)),
          refData(CreateTensor(refType, refData)),
          testcaseName(test_name) {}

    PartialShape inputShape;
    element::Type inType;
    element::Type refType;
    runtime::Tensor inputData;
    runtime::Tensor refData;
    std::string testcaseName;
};

class ReferenceConstantLayerTest : public testing::TestWithParam<ConstantParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        std::tie(function, inputData, refOutData) = CreateFunction(GetParam());
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

protected:
    using ReferenceType = std::tuple<std::shared_ptr<Function>, std::vector<ov::runtime::Tensor>, std::vector<ov::runtime::Tensor>>;

    virtual ReferenceType CreateFunction(const ParamType& params) {
        ReferenceType reference;
        auto A = op::v0::Constant::create(params.inType, params.inputShape.to_shape(), params.inputData.data());
        auto function = std::make_shared<Function>(A, ParameterVector{});
        reference = {function, {}, {params.refData}};
        return reference;
    }
};

class ReferenceConstantLayerTest_2Constant : public ReferenceConstantLayerTest {
protected:
    ReferenceType CreateFunction(const ParamType& params) override {
        ReferenceType reference;
        auto A = op::v0::Constant::create(params.inType, params.inputShape.to_shape(), params.inputData.data());
        auto B = op::v0::Constant::create(params.inType, params.inputShape.to_shape(), params.inputData.data());
        auto function = std::make_shared<Function>(NodeVector{A, B}, ParameterVector{});
        reference = {function, {}, {params.refData, params.refData}};
        return reference;
    }
};

class ReferenceConstantLayerTest_WithOp : public ReferenceConstantLayerTest {
protected:
    ReferenceType CreateFunction(const ParamType& params) override {
        ReferenceType reference;
        auto A = op::v0::Constant::create(params.inType, params.inputShape.to_shape(), params.inputData.data());
        auto function = std::make_shared<Function>(std::make_shared<op::v0::Abs>(A), ParameterVector{});
        reference = {function, {}, {params.refData}};
        return reference;
    }
};

class ReferenceConstantLayerTest_MultiUse : public ReferenceConstantLayerTest {
protected:
    ReferenceType CreateFunction(const ParamType& params) override {
        ReferenceType reference;
        const auto A = std::make_shared<op::v0::Constant>(
            params.inType,
            params.inputShape.to_shape(),
            std::vector<std::string>{std::to_string(*reinterpret_cast<int*>(params.inputData.data()))});
        auto function = std::make_shared<Function>(A, ParameterVector {});
        reference = {function, {}, {params.refData}};
        return reference;
    }
};

class ReferenceConstantLayerTest_EqualityBool : public ReferenceConstantLayerTest {
protected:
    ReferenceType CreateFunction(const ParamType& params) override {
        ReferenceType reference;
        auto A = op::v0::Constant::create(params.inType, params.inputShape.to_shape(), params.inputData.data());
        auto B = op::v0::Constant::create(params.inType, params.inputShape.to_shape(), {true, true, true, true});
        auto function = std::make_shared<Function>(std::make_shared<op::v1::Equal>(A, B), ParameterVector{});
        reference = {function, {}, {params.refData}};
        return reference;
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
    std::vector<ConstantParams> constantParams {
        // tensor_constant
        ConstantParams({2, 2, 2}, IN_ET, IN_ET,
                       std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                       std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                       "tensor_constant"),
    };
    return constantParams;
}

std::vector<ConstantParams> generateConstantDefinedTypeParams() {
    std::vector<ConstantParams> constantParams {
        // scalar_constant_float32
        ConstantParams({}, element::Type_t::f32, element::Type_t::f32,
            std::vector<float>{4.75},
            std::vector<float>{4.75f},
            "scalar_constant_float32"),
        // scalar_constant_int64
        ConstantParams({}, element::Type_t::i64, element::Type_t::i64,
            std::vector<int64_t>{0x4000000000000001},
            std::vector<int64_t>{0x4000000000000001},
            "scalar_constant_int64"),
        // tensor_constant_float32
        ConstantParams({2, 2}, element::Type_t::f32, element::Type_t::f32,
            std::vector<float>{4.75, 4.5, -5.25, 0.0},
            std::vector<float>{4.75f, 4.5f, -5.25f, 0.0f},
            "tensor_constant_float32"),
        // tensor_constant_int64
        ConstantParams({2}, element::Type_t::i64, element::Type_t::i64,
            std::vector<int64_t>{0x4000000000000001, 0x4000000000000002},
            std::vector<int64_t>{0x4000000000000001, 0x4000000000000002},
            "tensor_constant_int64"),
        // constant_equality_u4_2x2x3
        ConstantParams({2, 2, 3}, element::Type_t::u4, element::Type_t::u4,
            std::vector<char>{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xF, 0xF},
            std::vector<char>{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xF, 0xF},
            "constant_equality_u4_2x2x3"),
        // constant_equality_u4_1x3
        ConstantParams({1, 3}, element::Type_t::u4, element::Type_t::u4,
            std::vector<char>{0x1, 0x2, 0x3},
            std::vector<char>{0x1, 0x2, 0x3},
            "constant_equality_u4_1x3"),
        // constant_equality_u1_1x10
        ConstantParams({1, 10}, element::Type_t::u1, element::Type_t::u1,
            std::vector<char>{0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0},
            std::vector<char>{0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0},
            "constant_equality_u1_1x10"),
        // constant_equality_i4_2x2x3
        ConstantParams({2, 2, 3}, element::Type_t::i4, element::Type_t::i4,
            std::vector<char>{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xF, 0xF},
            std::vector<char>{0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xF, 0xF},
            "constant_equality_i4_2x2x3"),
        // constant_equality_i4_1x3
        ConstantParams({1, 3}, element::Type_t::i4, element::Type_t::i4,
            std::vector<char>{0x1, 0x2, 0x3},
            std::vector<char>{0x1, 0x2, 0x3},
            "constant_equality_i4_1x3"),
    };
    return constantParams;
}

std::vector<ConstantParams> generateConstantCombinedParams() {
    const std::vector<std::vector<ConstantParams>> constantTypeParams {
        generateConstantParams<element::Type_t::i4>(),
        generateConstantParams<element::Type_t::i8>(),
        generateConstantParams<element::Type_t::i16>(),
        generateConstantParams<element::Type_t::i32>(),
        generateConstantParams<element::Type_t::i64>(),
        generateConstantParams<element::Type_t::u1>(),
        generateConstantParams<element::Type_t::u4>(),
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
    const std::vector<std::vector<ConstantParams>> constantTypeParams {
        generateConstantParams<element::Type_t::i4>(),
        generateConstantParams<element::Type_t::i8>(),
        generateConstantParams<element::Type_t::i16>(),
        generateConstantParams<element::Type_t::i32>(),
        generateConstantParams<element::Type_t::i64>(),
        generateConstantParams<element::Type_t::u1>(),
        generateConstantParams<element::Type_t::u4>(),
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
    std::vector<ConstantParams> constantParams {
        // tensor_constant_with_op
        ConstantParams({2, 2, 2}, IN_ET, IN_ET,
                       std::vector<T>{-1, 2, 3, -4, 5, -6, -7, 8},
                       std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                       "tensor_constant_with_op"),
    };
    return constantParams;
}

std::vector<ConstantParams> generateConstantWithOpCombinedParams() {
    const std::vector<std::vector<ConstantParams>> constantTypeParams {
        generateConstantWithOpParams<element::Type_t::i4>(),
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
    const std::vector<ConstantParams> combinedParams {
        // constant_multi_use
        ConstantParams({}, element::Type_t::i32, element::Type_t::i32,
            std::vector<int32_t>{388},
            std::vector<int32_t>{388},
            "constant_multi_use"),
    };
    return combinedParams;
}

std::vector<ConstantParams> generateConstantDefinedTypeEqualityBoolCombinedParams() {
    const std::vector<ConstantParams> combinedParams {
        // constant_equality_bool
        ConstantParams({4}, element::Type_t::boolean, element::Type_t::boolean,
            std::vector<char>{true, false, true, false},
            std::vector<char>{true, false, true, false},
            "constant_equality_bool"),
    };
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs, ReferenceConstantLayerTest,
    testing::ValuesIn(generateConstantCombinedParams()), ReferenceConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs, ReferenceConstantLayerTest_2Constant,
    testing::ValuesIn(generateConstant2ConstantCombinedParams()), ReferenceConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs, ReferenceConstantLayerTest_WithOp,
    testing::ValuesIn(generateConstantWithOpCombinedParams()), ReferenceConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs, ReferenceConstantLayerTest_MultiUse,
    testing::ValuesIn(generateConstantDefinedTypeMultiUseCombinedParams()), ReferenceConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs, ReferenceConstantLayerTest_EqualityBool,
    testing::ValuesIn(generateConstantDefinedTypeEqualityBoolCombinedParams()), ReferenceConstantLayerTest::getTestCaseName);
} // namespace
