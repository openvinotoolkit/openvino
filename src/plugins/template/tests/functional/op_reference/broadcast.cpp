// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/broadcast.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reverse.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct BroadcastParams {
    BroadcastParams(const reference_tests::Tensor& dataTensor,
                    const reference_tests::Tensor& targetShapeTensor,
                    const reference_tests::Tensor& expectedTensor,
                    const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          targetShapeTensor(targetShapeTensor),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor targetShapeTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
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
    static std::shared_ptr<Model> CreateFunction(const BroadcastParams& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto f = std::make_shared<Model>(
            std::make_shared<op::v1::Broadcast>(A,
                                                op::v0::Constant::create(params.targetShapeTensor.type,
                                                                         params.targetShapeTensor.shape,
                                                                         params.targetShapeTensor.data.data())),
            ParameterVector{A});
        return f;
    }
};

TEST_P(ReferenceBroadcastTest, CompareWithRefs) {
    Exec();
}

class ReferenceBroadcastTestV3 : public ReferenceBroadcastTest {
private:
    static std::shared_ptr<Model> CreateFunction(const BroadcastParams& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto f = std::make_shared<Model>(
            std::make_shared<op::v3::Broadcast>(A,
                                                op::v0::Constant::create(params.targetShapeTensor.type,
                                                                         params.targetShapeTensor.shape,
                                                                         params.targetShapeTensor.data.data())),
            ParameterVector{A});
        return f;
    }
};

TEST_P(ReferenceBroadcastTestV3, CompareWithRefs) {
    Exec();
}

struct BroadcastParamsExplicitAxis : BroadcastParams {
    BroadcastParamsExplicitAxis(const reference_tests::Tensor& dataTensor,
                                const reference_tests::Tensor& targetShapeTensor,
                                const reference_tests::Tensor& axesMappingTensor,
                                const reference_tests::Tensor& expectedTensor,
                                const std::string& testcaseName = "")
        : BroadcastParams(dataTensor, targetShapeTensor, expectedTensor, testcaseName),
          axesMappingTensor(axesMappingTensor) {}

    reference_tests::Tensor axesMappingTensor;
};

class ReferenceBroadcastTestExplicitAxis : public testing::TestWithParam<BroadcastParamsExplicitAxis>,
                                           public CommonReferenceTest {
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
    static std::shared_ptr<Model> CreateFunction(const BroadcastParamsExplicitAxis& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto f = std::make_shared<Model>(
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

TEST_P(ReferenceBroadcastTestExplicitAxis, CompareWithRefs) {
    Exec();
}

struct BroadcastParamsTestHelper {
    BroadcastParamsTestHelper(const Shape& shapeA,
                              const Shape& shapeR,
                              const AxisSet& axes,
                              const std::string& testcaseName = "")
        : shapeA(shapeA),
          shapeR(shapeR),
          axes(axes),
          testcaseName(testcaseName) {}

    Shape shapeA;
    Shape shapeR;
    AxisSet axes;
    std::string testcaseName;
};

class ReferenceBroadcastTestTestHelper : public testing::TestWithParam<BroadcastParamsTestHelper>,
                                         public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        std::vector<float> inpData(shape_size<const Shape>(params.shapeA));
        iota(inpData.begin(), inpData.end(), 1.f);
        const auto refA = CreateTensor(params.shapeA, element::f32, inpData);
        inputData = {refA};
    }

    void SetUp1() {
        auto params = GetParam();
        function = CreateFunction(params);
        std::vector<float> inpData(shape_size<const Shape>(params.shapeA));
        iota(inpData.begin(), inpData.end(), 1.f);
        const auto wrkA = CreateTensor(params.shapeA, element::f32, inpData);
        inputData = {wrkA};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BroadcastParamsTestHelper>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aShape=" << param.shapeA;
        result << "_rShape=" << param.shapeR;
        if (param.testcaseName != "") {
            result << "_axes=" << param.axes;
            result << "_=" << param.testcaseName;
        } else {
            result << "_axes=" << param.axes;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const BroadcastParamsTestHelper& params) {
        const auto A = std::make_shared<op::v0::Parameter>(element::f32, params.shapeA);
        const auto shape_const = op::v0::Constant::create(element::u64, Shape{params.shapeR.size()}, params.shapeR);
        std::shared_ptr<Node> broadcast;
        if (params.axes.size() > 0) {
            auto axes_const =
                op::v0::Constant::create(element::i64, Shape{params.axes.size()}, params.axes.to_vector());
            broadcast = std::make_shared<op::v1::Broadcast>(A, shape_const, axes_const);
        } else {
            broadcast = std::make_shared<op::v1::Broadcast>(A, shape_const);
        }
        auto f = std::make_shared<Model>(broadcast, ParameterVector{A});
        return f;
    }

protected:
    void GenerateRefOutData() {
        actualOutData.clear();
        for (const auto& output : executableNetwork.outputs()) {
            actualOutData.emplace_back(inferRequest.get_tensor(output));
        }
        refOutData = actualOutData;
    }
};

TEST_P(ReferenceBroadcastTestTestHelper, CompareWithRefs) {
    LoadNetwork();
    FillInputs();
    Infer();
    GenerateRefOutData();
    SetUp1();
    Exec();
}

class ReferenceBroadcastTestExplicitAxisReversed : public ReferenceBroadcastTestExplicitAxis {
private:
    static std::shared_ptr<Model> CreateFunction(const BroadcastParamsExplicitAxis& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        auto broadcast =
            std::make_shared<op::v1::Broadcast>(A,
                                                op::v0::Constant::create(params.targetShapeTensor.type,
                                                                         params.targetShapeTensor.shape,
                                                                         params.targetShapeTensor.data.data()),
                                                op::v0::Constant::create(params.axesMappingTensor.type,
                                                                         params.axesMappingTensor.shape,
                                                                         params.axesMappingTensor.data.data()));
        auto reverse = std::make_shared<op::v1::Reverse>(broadcast,
                                                         op::v0::Constant::create(element::i64, {1}, {1}),
                                                         op::v1::Reverse::Mode::INDEX);
        auto f = std::make_shared<Model>(NodeVector{reverse}, ParameterVector{A});
        return f;
    }
};

TEST_P(ReferenceBroadcastTestExplicitAxisReversed, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<BroadcastParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<BroadcastParams> params{
        BroadcastParams(reference_tests::Tensor(ET, {}, std::vector<T>{6}),
                        reference_tests::Tensor(element::u64, {1}, std::vector<uint64_t>{4}),
                        reference_tests::Tensor(ET, {4}, std::vector<T>{6, 6, 6, 6}),
                        "broadcast_scalar_vector"),
        BroadcastParams(reference_tests::Tensor(ET, {}, std::vector<T>{6}),
                        reference_tests::Tensor(element::u64, {2}, std::vector<uint64_t>{2, 2}),
                        reference_tests::Tensor(ET, {2, 2}, std::vector<T>{6, 6, 6, 6}),
                        "broadcast_scalar_matrix"),
        BroadcastParams(reference_tests::Tensor(ET, {}, std::vector<T>{6}),
                        reference_tests::Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
                        reference_tests::Tensor(ET, {2, 2, 2}, std::vector<T>{6, 6, 6, 6, 6, 6, 6, 6}),
                        "broadcast_scalar_tensor"),
        BroadcastParams(reference_tests::Tensor(ET, {2, 2, 2}, std::vector<T>{2, 4, 6, 8, 16, 32, 64, 127}),
                        reference_tests::Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
                        reference_tests::Tensor(ET, {2, 2, 2}, std::vector<T>{2, 4, 6, 8, 16, 32, 64, 127}),
                        "broadcast_trivial"),
        BroadcastParams(reference_tests::Tensor(ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
                        reference_tests::Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
                        reference_tests::Tensor(ET, {2, 2, 2}, std::vector<T>{1, 2, 3, 4, 1, 2, 3, 4}),
                        "broadcast_matrix_0"),
    };
    return params;
}

std::vector<BroadcastParams> generateCombinedParams() {
    const std::vector<std::vector<BroadcastParams>> generatedParams{
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

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs,
                         ReferenceBroadcastTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceBroadcastTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs,
                         ReferenceBroadcastTestV3,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceBroadcastTest::getTestCaseName);

template <element::Type_t ET>
std::vector<BroadcastParamsExplicitAxis> generateParamsExplicitAxis() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<BroadcastParamsExplicitAxis> params{
        BroadcastParamsExplicitAxis(reference_tests::Tensor(ET, {}, std::vector<T>{6}),
                                    reference_tests::Tensor(element::u64, {2}, std::vector<uint64_t>{1, 2}),
                                    reference_tests::Tensor(element::i64, {1}, std::vector<int64_t>{0}),
                                    reference_tests::Tensor(ET, {1, 2}, std::vector<T>{6, 6}),
                                    "broadcast_scalar_vector_explicit_axis_0"),
        BroadcastParamsExplicitAxis(
            reference_tests::Tensor(ET, {3}, std::vector<T>{1, 2, 3}),
            reference_tests::Tensor(element::u64, {2}, std::vector<uint64_t>{3, 4}),
            reference_tests::Tensor(element::i64, {1}, std::vector<int64_t>{0}),
            reference_tests::Tensor(ET, {3, 4}, std::vector<T>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}),
            "broadcast_vector_colwise"),
        BroadcastParamsExplicitAxis(
            reference_tests::Tensor(ET, {4}, std::vector<T>{1, 2, 3, 4}),
            reference_tests::Tensor(element::u64, {2}, std::vector<uint64_t>{3, 4}),
            reference_tests::Tensor(element::i64, {1}, std::vector<int64_t>{1}),
            reference_tests::Tensor(ET, {3, 4}, std::vector<T>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
            "broadcast_vector_rowwise"),
        BroadcastParamsExplicitAxis(reference_tests::Tensor(ET, {1}, std::vector<T>{4}),
                                    reference_tests::Tensor(element::u64, {2}, std::vector<uint64_t>{3, 1}),
                                    reference_tests::Tensor(element::i64, {1}, std::vector<int64_t>{1}),
                                    reference_tests::Tensor(ET, {3, 1}, std::vector<T>{4, 4, 4}),
                                    "broadcast_scalar_to_matrix"),
        BroadcastParamsExplicitAxis(reference_tests::Tensor(ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
                                    reference_tests::Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
                                    reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{0, 2}),
                                    reference_tests::Tensor(ET, {2, 2, 2}, std::vector<T>{1, 2, 1, 2, 3, 4, 3, 4}),
                                    "broadcast_matrix_1"),
        BroadcastParamsExplicitAxis(reference_tests::Tensor(ET, {2, 2}, std::vector<T>{1, 2, 3, 4}),
                                    reference_tests::Tensor(element::u64, {3}, std::vector<uint64_t>{2, 2, 2}),
                                    reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{0, 1}),
                                    reference_tests::Tensor(ET, {2, 2, 2}, std::vector<T>{1, 1, 2, 2, 3, 3, 4, 4}),
                                    "broadcast_matrix_2"),
    };
    return params;
}

std::vector<BroadcastParamsExplicitAxis> generateCombinedParamsExplicitAxis() {
    const std::vector<std::vector<BroadcastParamsExplicitAxis>> generatedParams{
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

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs,
                         ReferenceBroadcastTestExplicitAxis,
                         testing::ValuesIn(generateCombinedParamsExplicitAxis()),
                         ReferenceBroadcastTestExplicitAxis::getTestCaseName);

std::vector<BroadcastParamsTestHelper> generateParamsTestHelper() {
    std::vector<BroadcastParamsTestHelper> params{
        BroadcastParamsTestHelper({2}, {3, 2, 4}, {1}, "broadcast_algo_vector_middle"),
        BroadcastParamsTestHelper({2}, {3, 2}, {1}, "broadcast_algo_vector_forward_2"),
        BroadcastParamsTestHelper({2}, {4, 3, 2}, {2}, "broadcast_algo_vector_forward_3"),
        BroadcastParamsTestHelper({2}, {5, 4, 3, 2}, {3}, "broadcast_algo_vector_forward_4"),
        BroadcastParamsTestHelper({}, {5, 4, 3, 2}, {}, "broadcast_algo_scalar"),
        BroadcastParamsTestHelper({2}, {2, 3}, {0}, "broadcast_algo_vector_backward_2"),
        BroadcastParamsTestHelper({2}, {2, 3, 4}, {0}, "broadcast_algo_vector_backward_3"),
        BroadcastParamsTestHelper({2}, {2, 3, 4, 5}, {0}, "broadcast_algo_vector_backward_4"),
        BroadcastParamsTestHelper({4, 5}, {2, 3, 4, 5}, {2, 3}, "broadcast_algo_matrix_backward_4"),
        BroadcastParamsTestHelper({3, 5}, {2, 3, 4, 5}, {1, 3}, "broadcast_algo_matrix_stride_1"),
        BroadcastParamsTestHelper({3, 4}, {2, 3, 4, 5}, {1, 2}, "broadcast_algo_matrix_stride_2"),
        BroadcastParamsTestHelper({2, 4}, {2, 3, 4, 5}, {0, 2}, "broadcast_algo_matrix_stride_3"),
        BroadcastParamsTestHelper({2, 3, 4}, {5, 2, 3, 4}, {1, 2, 3}, "broadcast_algo_3d_backward"),
        BroadcastParamsTestHelper({2, 3, 4}, {2, 5, 3, 4}, {0, 2, 3}, "broadcast_algo_3d_stride_1"),
        BroadcastParamsTestHelper({2, 3, 4}, {2, 3, 5, 4}, {0, 1, 3}, "broadcast_algo_3d_stride_2"),
        BroadcastParamsTestHelper({3, 1}, {2, 3, 3}, {1, 2}, "broadcast_algo_3d_diffrent_rank"),
        BroadcastParamsTestHelper({2, 3, 1, 1}, {2, 3, 4, 5}, {0, 1, 2, 3}, "broadcast_algo_4d_same_rank"),
    };
    return params;
}

std::vector<BroadcastParamsTestHelper> generateCombinedParamsTestHelper() {
    const std::vector<std::vector<BroadcastParamsTestHelper>> generatedParams{
        generateParamsTestHelper(),
    };
    std::vector<BroadcastParamsTestHelper> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs,
                         ReferenceBroadcastTestTestHelper,
                         testing::ValuesIn(generateCombinedParamsTestHelper()),
                         ReferenceBroadcastTestTestHelper::getTestCaseName);

template <element::Type_t ET>
std::vector<BroadcastParamsExplicitAxis> generateParamsExplicitAxisReversed() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<BroadcastParamsExplicitAxis> params{
        BroadcastParamsExplicitAxis(
            reference_tests::Tensor(ET, {4}, std::vector<T>{1, 2, 3, 4}),
            reference_tests::Tensor(element::u64, {2}, std::vector<uint64_t>{3, 4}),
            reference_tests::Tensor(element::i64, {1}, std::vector<int64_t>{1}),
            reference_tests::Tensor(ET, {3, 4}, std::vector<T>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
            "broadcast_vector_rowwise_reversed"),
    };
    return params;
}

std::vector<BroadcastParamsExplicitAxis> generateCombinedParamsExplicitAxisReversed() {
    const std::vector<std::vector<BroadcastParamsExplicitAxis>> generatedParams{
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

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast_With_Hardcoded_Refs,
                         ReferenceBroadcastTestExplicitAxisReversed,
                         testing::ValuesIn(generateCombinedParamsExplicitAxisReversed()),
                         ReferenceBroadcastTestExplicitAxis::getTestCaseName);
}  // namespace
