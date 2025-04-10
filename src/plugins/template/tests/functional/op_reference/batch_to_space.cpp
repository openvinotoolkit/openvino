// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_to_space.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct BatchToSpaceParams {
    BatchToSpaceParams(const reference_tests::Tensor& dataTensor,
                       const reference_tests::Tensor& blockShapeTensor,
                       const reference_tests::Tensor& cropsBeginTensor,
                       const reference_tests::Tensor& cropsEndTensor,
                       const reference_tests::Tensor& expectedTensor,
                       const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          blockShapeTensor(blockShapeTensor),
          cropsBeginTensor(cropsBeginTensor),
          cropsEndTensor(cropsEndTensor),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor blockShapeTensor;
    reference_tests::Tensor cropsBeginTensor;
    reference_tests::Tensor cropsEndTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceBatchToSpaceLayerTest : public testing::TestWithParam<BatchToSpaceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data,
                     params.blockShapeTensor.data,
                     params.cropsBeginTensor.data,
                     params.cropsEndTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<BatchToSpaceParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_bsType=" << param.blockShapeTensor.type;
        result << "_bsShape=" << param.blockShapeTensor.shape;
        result << "_cbType=" << param.cropsBeginTensor.type;
        result << "_cbShape=" << param.cropsBeginTensor.shape;
        result << "_ceType=" << param.cropsEndTensor.type;
        result << "_ceShape=" << param.cropsEndTensor.shape;
        result << "_eType=" << param.expectedTensor.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedTensor.shape;
            result << "_" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedTensor.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const BatchToSpaceParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto blockShape = std::make_shared<op::v0::Parameter>(element::i64, params.blockShapeTensor.shape);
        const auto cropsBegin = std::make_shared<op::v0::Parameter>(element::i64, params.cropsBeginTensor.shape);
        const auto cropsEnd = std::make_shared<op::v0::Parameter>(element::i64, params.cropsEndTensor.shape);
        const auto batchToSpace = std::make_shared<op::v1::BatchToSpace>(data, blockShape, cropsBegin, cropsEnd);
        return std::make_shared<Model>(NodeVector{batchToSpace},
                                       ParameterVector{data, blockShape, cropsBegin, cropsEnd});
    }
};

TEST_P(ReferenceBatchToSpaceLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<BatchToSpaceParams> generateBatchToSpaceParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<BatchToSpaceParams> batchToSpaceParams{
        // input_with_shape_4x3
        BatchToSpaceParams(
            reference_tests::Tensor({4, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor({2}, element::i64, std::vector<int64_t>{1, 2}),
            reference_tests::Tensor({2}, element::i64, std::vector<int64_t>{0, 0}),
            reference_tests::Tensor({2}, element::i64, std::vector<int64_t>{0, 0}),
            reference_tests::Tensor({2, 6}, IN_ET, std::vector<T>{1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12}),
            "input_with_shape_4x3"),

        // input_with_shape_4x1x3
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor({3}, element::i64, std::vector<int64_t>{1, 1, 2}),
            reference_tests::Tensor({3}, element::i64, std::vector<int64_t>{0, 0, 0}),
            reference_tests::Tensor({3}, element::i64, std::vector<int64_t>{0, 0, 0}),
            reference_tests::Tensor({2, 1, 6}, IN_ET, std::vector<T>{1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12}),
            "input_with_shape_4x1x3"),

        // input_with_shape_4x1x1x3
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 1, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 1, 2}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({2, 1, 1, 6}, IN_ET, std::vector<T>{1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12}),
            "input_with_shape_4x1x1x3"),
        // input_with_shape_4x1x1x3_1
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 1, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 2, 1}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({2, 1, 2, 3}, IN_ET, std::vector<T>{1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}),
            "input_with_shape_4x1x1x3_1"),
        // input_with_shape_4x1x1x3_2
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 1, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 2, 2}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({1, 1, 2, 6}, IN_ET, std::vector<T>{1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12}),
            "input_with_shape_4x1x1x3_2"),

        // input_with_shape_4x1x2x3
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 2, 3}, IN_ET, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 1, 2}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({2, 1, 2, 6}, IN_ET, std::vector<T>{1, 13, 2, 14, 3, 15, 4,  16, 5,  17, 6,  18,
                                                                        7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24}),
            "input_with_shape_4x1x2x3"),
        // input_with_shape_4x1x2x3_1
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 2, 3}, IN_ET, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 2, 1}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({2, 1, 4, 3}, IN_ET, std::vector<T>{1, 2, 3, 13, 14, 15, 4,  5,  6,  16, 17, 18,
                                                                        7, 8, 9, 19, 20, 21, 10, 11, 12, 22, 23, 24}),
            "input_with_shape_4x1x2x3_1"),
        // input_with_shape_4x1x2x3_2
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 2, 3}, IN_ET, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 2, 2}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({1, 1, 4, 6}, IN_ET, std::vector<T>{1, 7,  2, 8,  3, 9,  13, 19, 14, 20, 15, 21,
                                                                        4, 10, 5, 11, 6, 12, 16, 22, 17, 23, 18, 24}),
            "input_with_shape_4x1x2x3_2"),

        // input_with_shape_with_crop_4x1x2x3
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 2, 3}, IN_ET, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 2, 2}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 2}),
            reference_tests::Tensor({1, 1, 4, 4},
                                    IN_ET,
                                    std::vector<T>{1, 7, 2, 8, 13, 19, 14, 20, 4, 10, 5, 11, 16, 22, 17, 23}),
            "input_with_shape_with_crop_4x1x2x3"),
        // input_with_shape_with_crop_4x1x2x3_1
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 2, 3}, IN_ET, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 2, 2}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 2}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
            reference_tests::Tensor({1, 1, 4, 4},
                                    IN_ET,
                                    std::vector<T>{2, 8, 3, 9, 14, 20, 15, 21, 5, 11, 6, 12, 17, 23, 18, 24}),
            "input_with_shape_with_crop_4x1x2x3_1"),
        // input_with_shape_with_crop_4x1x2x3_2
        BatchToSpaceParams(
            reference_tests::Tensor({4, 1, 2, 3}, IN_ET, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 2, 2}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 1, 0}),
            reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 1, 0}),
            reference_tests::Tensor({1, 1, 2, 6}, IN_ET, std::vector<T>{13, 19, 14, 20, 15, 21, 4, 10, 5, 11, 6, 12}),
            "input_with_shape_with_crop_4x1x2x3_2"),
    };
    return batchToSpaceParams;
}

std::vector<BatchToSpaceParams> generateBatchToSpaceCombinedParams() {
    const std::vector<std::vector<BatchToSpaceParams>> batchToSpaceTypeParams{
        generateBatchToSpaceParams<element::Type_t::i8>(),
        generateBatchToSpaceParams<element::Type_t::i16>(),
        generateBatchToSpaceParams<element::Type_t::i32>(),
        generateBatchToSpaceParams<element::Type_t::i64>(),
        generateBatchToSpaceParams<element::Type_t::u8>(),
        generateBatchToSpaceParams<element::Type_t::u16>(),
        generateBatchToSpaceParams<element::Type_t::u32>(),
        generateBatchToSpaceParams<element::Type_t::u64>(),
        generateBatchToSpaceParams<element::Type_t::bf16>(),
        generateBatchToSpaceParams<element::Type_t::f16>(),
        generateBatchToSpaceParams<element::Type_t::f32>(),
        generateBatchToSpaceParams<element::Type_t::f64>(),
    };
    std::vector<BatchToSpaceParams> combinedParams;

    for (const auto& params : batchToSpaceTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_BatchToSpace_With_Hardcoded_Refs,
                         ReferenceBatchToSpaceLayerTest,
                         testing::ValuesIn(generateBatchToSpaceCombinedParams()),
                         ReferenceBatchToSpaceLayerTest::getTestCaseName);
}  // namespace
