// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reverse.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct BroadcastParams {
    BroadcastParams(
        const Tensor& dataTensor, const Tensor& targetShapeTensor,
        const Tensor& expectedTensor, const std::string& testcaseName = "") :
        dataTensor(dataTensor), targetShapeTensor(targetShapeTensor),
        expectedTensor(expectedTensor), testcaseName(testcaseName) {}

    Tensor dataTensor;
    Tensor targetShapeTensor;
    Tensor expectedTensor;
    std::string testcaseName;
};

struct BroadcastParamsExplicitAxis : BroadcastParams {
    BroadcastParamsExplicitAxis(
        const Tensor& dataTensor, const Tensor& targetShapeTensor, const Tensor& axesMappingTensor,
        const Tensor& expectedTensor, const std::string& testcaseName = "") :
        BroadcastParams(dataTensor, targetShapeTensor, expectedTensor, testcaseName),
        axesMappingTensor(axesMappingTensor) {}

    Tensor axesMappingTensor;
};

class ReferenceBroadcastTest : public testing::TestWithParam<BroadcastParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_tsType=" << param.targetShapeTensor.type;
        result << "_tsShape=" << param.targetShapeTensor.shape;
        result << "_eType=" << param.expectedTensor.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensor.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const BroadcastParams& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto f = std::make_shared<Function>(
            std::make_shared<op::v1::Broadcast>(A, op::v0::Constant::create(params.targetShapeTensor.type,
                                                                            params.targetShapeTensor.shape,
                                                                            params.targetShapeTensor.data.data())),
            ParameterVector{A});
        return f;
    }
};

class ReferenceBroadcastTestV3 : public ReferenceBroadcastTest {
private:
    static std::shared_ptr<Function> CreateFunction(const BroadcastParams& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto f = std::make_shared<Function>(
            std::make_shared<op::v3::Broadcast>(A, op::v0::Constant::create(params.targetShapeTensor.type,
                                                                            params.targetShapeTensor.shape,
                                                                            params.targetShapeTensor.data.data())),
            ParameterVector{A});
        return f;
    }
};

class ReferenceBroadcastTestExplicitAxis : public testing::TestWithParam<BroadcastParamsExplicitAxis>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastParamsExplicitAxis>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_tsType=" << param.targetShapeTensor.type;
        result << "_tsShape=" << param.targetShapeTensor.shape;
        result << "_amType=" << param.axesMappingTensor.type;
        result << "_amShape=" << param.axesMappingTensor.shape;
        result << "_eType=" << param.expectedTensor.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensor.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const BroadcastParamsExplicitAxis& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto f = std::make_shared<Function>(
            std::make_shared<op::v1::Broadcast>(A,
                                                op::v0::Constant::create(params.targetShapeTensor.type,
                                                                         params.targetShapeTensor.shape,
                                                                         params.targetShapeTensor.data.data()),
                                                op::v0::Constant::create(params.axesMappingTensor.type,
                                                                         params.axesMappingTensor.shape,
                                                                         params.axesMappingTensor.data.data())),
            ParameterVector{A});
        return f;
    }
};

class ReferenceBroadcastTestExplicitAxisReversed : public ReferenceBroadcastTestExplicitAxis {
private:
    static std::shared_ptr<Function> CreateFunction(const BroadcastParamsExplicitAxis& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        auto broadcast = std::make_shared<op::v1::Broadcast>(
            A,
            op::v0::Constant::create(params.targetShapeTensor.type,
                                     params.targetShapeTensor.shape,
                                     params.targetShapeTensor.data.data()),
            op::v0::Constant::create(params.axesMappingTensor.type,
                                     params.axesMappingTensor.shape,
                                     params.axesMappingTensor.data.data()));
        auto reverse = std::make_shared<op::v1::Reverse>(broadcast,
                                                         op::v0::Constant::create(element::i64, {1}, {1}),
                                                         op::v1::Reverse::Mode::INDEX);
        auto f = std::make_shared<Function>(NodeVector{reverse}, ParameterVector{A});
        return f;
    }
};

TEST_P(ReferenceBroadcastTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceBroadcastTestV3, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceBroadcastTestExplicitAxis, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceBroadcastTestExplicitAxisReversed, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<BroadcastParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<BroadcastParams> params {
        BroadcastParams(
            Tensor(ET, {}, std::vector<T>{6}),
            Tensor(element::u64, {1}, std::vector<uint64_t>{4}),
            Tensor(ET, {4}, std::vector<T>{6, 6, 6, 6}),
            "broadcast_scalar_vector"),
        BroadcastParams(
            Tensor(ET, {}, std::vector<T>{6}),
            Tensor(element::u64, {2}, std::vector<uint64_t>{2, 2}),
            Tensor(ET, {2, 2}, std::vector<T>{6, 6, 6, 6}),
            "broadcast_scalar_matrix"),
        BroadcastParams(
            Tensor(ET, {}, std::vector<T>{6}),
            Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
            Tensor(ET, {2, 2, 2}, std::vector<T>{6, 6, 6, 6, 6, 6, 6, 6}),
            "broadcast_scalar_tensor"),
        BroadcastParams(
            Tensor(ET, {2, 2, 2}, std::vector<T>{2, 4, 6, 8, 16, 32, 64, 127}),
            Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
            Tensor(ET, {2, 2, 2}, std::vector<T>{2, 4, 6, 8, 16, 32, 64, 127}),
            "broadcast_trivial"),
        BroadcastParams(
            Tensor(ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
            Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
            Tensor(ET, {2, 2, 2}, std::vector<T>{1, 2, 3, 4, 1, 2, 3, 4}),
            "broadcast_matrix_0"),
    };
    return params;
}

std::vector<BroadcastParams> generateCombinedParams() {
    const std::vector<std::vector<BroadcastParams>> generatedParams {
        generateParams<element::Type_t::i8>(),
        generateParams<element::Type_t::i16>(),
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::i64>(),
        generateParams<element::Type_t::u8>(),
        generateParams<element::Type_t::u16>(),
        generateParams<element::Type_t::u32>(),
        generateParams<element::Type_t::u64>(),
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<BroadcastParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<BroadcastParamsExplicitAxis> generateParamsExplicitAxis() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<BroadcastParamsExplicitAxis> params {
        BroadcastParamsExplicitAxis(
            Tensor(ET, {}, std::vector<T>{6}),
            Tensor(element::u64, {2}, std::vector<uint64_t>{1, 2}),
            Tensor(element::i64, {1}, std::vector<int64_t>{0}),
            Tensor(ET, {1, 2}, std::vector<T>{6, 6}),
            "broadcast_scalar_vector_explicit_axis_0"),
        BroadcastParamsExplicitAxis(
            Tensor(ET, {3}, std::vector<T>{1, 2, 3}),
            Tensor(element::u64, {2}, std::vector<uint64_t>{3, 4}),
            Tensor(element::i64, {1}, std::vector<int64_t>{0}),
            Tensor(ET, {3, 4}, std::vector<T>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}),
            "broadcast_vector_colwise"),
        BroadcastParamsExplicitAxis(
            Tensor(ET, {4}, std::vector<T>{1, 2, 3, 4}),
            Tensor(element::u64, {2}, std::vector<uint64_t>{3, 4}),
            Tensor(element::i64, {1}, std::vector<int64_t>{1}),
            Tensor(ET, {3, 4}, std::vector<T>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
            "broadcast_vector_rowwise"),
        BroadcastParamsExplicitAxis(
            Tensor(ET, {1}, std::vector<T>{4}),
            Tensor(element::u64, {2}, std::vector<uint64_t>{3, 1}),
            Tensor(element::i64, {1}, std::vector<int64_t>{1}),
            Tensor(ET, {3, 1}, std::vector<T>{4, 4, 4}),
            "broadcast_scalar_to_matrix"),
        BroadcastParamsExplicitAxis(
            Tensor(ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
            Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
            Tensor(element::i64, {2}, std::vector<int64_t>{0, 2}),
            Tensor(ET, {2, 2, 2}, std::vector<T>{1, 2, 1, 2, 3, 4, 3, 4}),
            "broadcast_matrix_1"),
        BroadcastParamsExplicitAxis(
            Tensor(ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
            Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
            Tensor(element::i64, {2}, std::vector<int64_t>{0, 1}),
            Tensor(ET, {2, 2, 2}, std::vector<T>{1, 1, 2, 2, 3, 3, 4, 4}),
            "broadcast_matrix_2"),
    };
    return params;
}

std::vector<BroadcastParamsExplicitAxis> generateCombinedParamsExplicitAxis() {
    const std::vector<std::vector<BroadcastParamsExplicitAxis>> generatedParams {
        generateParamsExplicitAxis<element::Type_t::i8>(),
        generateParamsExplicitAxis<element::Type_t::i16>(),
        generateParamsExplicitAxis<element::Type_t::i32>(),
        generateParamsExplicitAxis<element::Type_t::i64>(),
        generateParamsExplicitAxis<element::Type_t::u8>(),
        generateParamsExplicitAxis<element::Type_t::u16>(),
        generateParamsExplicitAxis<element::Type_t::u32>(),
        generateParamsExplicitAxis<element::Type_t::u64>(),
        generateParamsExplicitAxis<element::Type_t::bf16>(),
        generateParamsExplicitAxis<element::Type_t::f16>(),
        generateParamsExplicitAxis<element::Type_t::f32>(),
        generateParamsExplicitAxis<element::Type_t::f64>(),
    };
    std::vector<BroadcastParamsExplicitAxis> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<BroadcastParamsExplicitAxis> generateParamsExplicitAxisReversed() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<BroadcastParamsExplicitAxis> params {
        BroadcastParamsExplicitAxis(
            Tensor(ET, {4}, std::vector<T>{1, 2, 3, 4}),
            Tensor(element::u64, {2}, std::vector<uint64_t>{3, 4}),
            Tensor(element::i64, {1}, std::vector<int64_t>{1}),
            Tensor(ET, {3, 4}, std::vector<T>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
            "broadcast_vector_rowwise_reversed"),
    };
    return params;
}

std::vector<BroadcastParamsExplicitAxis> generateCombinedParamsExplicitAxisReversed() {
    const std::vector<std::vector<BroadcastParamsExplicitAxis>> generatedParams {
        generateParamsExplicitAxisReversed<element::Type_t::i8>(),
        generateParamsExplicitAxisReversed<element::Type_t::i16>(),
        generateParamsExplicitAxisReversed<element::Type_t::i32>(),
        generateParamsExplicitAxisReversed<element::Type_t::i64>(),
        generateParamsExplicitAxisReversed<element::Type_t::u8>(),
        generateParamsExplicitAxisReversed<element::Type_t::u16>(),
        generateParamsExplicitAxisReversed<element::Type_t::u32>(),
        generateParamsExplicitAxisReversed<element::Type_t::u64>(),
        generateParamsExplicitAxisReversed<element::Type_t::bf16>(),
        generateParamsExplicitAxisReversed<element::Type_t::f16>(),
        generateParamsExplicitAxisReversed<element::Type_t::f32>(),
        generateParamsExplicitAxisReversed<element::Type_t::f64>(),
    };
    std::vector<BroadcastParamsExplicitAxis> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs, ReferenceBroadcastTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceBroadcastTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs, ReferenceBroadcastTestV3,
    testing::ValuesIn(generateCombinedParams()), ReferenceBroadcastTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs, ReferenceBroadcastTestExplicitAxis,
    testing::ValuesIn(generateCombinedParamsExplicitAxis()), ReferenceBroadcastTestExplicitAxis::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs, ReferenceBroadcastTestExplicitAxisReversed,
    testing::ValuesIn(generateCombinedParamsExplicitAxisReversed()), ReferenceBroadcastTestExplicitAxis::getTestCaseName);
} // namespace