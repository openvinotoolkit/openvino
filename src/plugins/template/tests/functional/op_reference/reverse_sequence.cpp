// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse_sequence.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ReverseSequenceParams {
    ReverseSequenceParams(const int64_t batchAxis,
                          const int64_t seqAxis,
                          const reference_tests::Tensor& dataTensor,
                          const reference_tests::Tensor& seqLengthsTensor,
                          const reference_tests::Tensor& expectedTensor)
        : mBatchAxis(batchAxis),
          mSeqAxis(seqAxis),
          mDataTensor(dataTensor),
          mSeqLengthsTensor(seqLengthsTensor),
          mExpectedTensor(expectedTensor) {}
    int64_t mBatchAxis;
    int64_t mSeqAxis;
    reference_tests::Tensor mDataTensor;
    reference_tests::Tensor mSeqLengthsTensor;
    reference_tests::Tensor mExpectedTensor;
};

class ReferenceReverseSequenceTest : public testing::TestWithParam<ReverseSequenceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.mDataTensor.data, params.mSeqLengthsTensor.data};
        refOutData = {params.mExpectedTensor.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ReverseSequenceParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dataType=" << param.mDataTensor.type << "_";
        result << "dataShape=" << param.mDataTensor.shape << "_";
        result << "seqLenType=" << param.mSeqLengthsTensor.type << "_";
        result << "seqLenShape=" << param.mSeqLengthsTensor.shape << "_";
        result << "batchAxis=" << param.mBatchAxis << "_";
        result << "seqAxis=" << param.mSeqAxis;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ReverseSequenceParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.mDataTensor.type, params.mDataTensor.shape);
        const auto seqLengths =
            std::make_shared<op::v0::Parameter>(params.mSeqLengthsTensor.type, params.mSeqLengthsTensor.shape);
        const auto reverseSequence =
            std::make_shared<op::v0::ReverseSequence>(data, seqLengths, params.mBatchAxis, params.mSeqAxis);
        return std::make_shared<ov::Model>(NodeVector{reverseSequence}, ParameterVector{data, seqLengths});
    }
};

TEST_P(ReferenceReverseSequenceTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ReverseSequenceParams> generateReverseSeqParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ReverseSequenceParams> reverseSeqParams{
        // 2D
        ReverseSequenceParams(
            1,
            0,
            reference_tests::Tensor({4, 4},
                                    IN_ET,
                                    std::vector<T>{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}),
            reference_tests::Tensor({4}, element::i32, std::vector<int32_t>{4, 3, 2, 1}),
            reference_tests::Tensor({4, 4},
                                    IN_ET,
                                    std::vector<T>{3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15})),
        ReverseSequenceParams(
            0,
            1,
            reference_tests::Tensor({4, 4},
                                    IN_ET,
                                    std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
            reference_tests::Tensor({4}, element::i32, std::vector<int32_t>{1, 2, 3, 4}),
            reference_tests::Tensor({4, 4},
                                    IN_ET,
                                    std::vector<T>{0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12})),
        // 4D
        ReverseSequenceParams(
            2,
            1,
            reference_tests::Tensor({2, 3, 4, 2}, IN_ET, std::vector<T>{0,  0, 3,  0, 6,  0, 9,  0, 1,  0, 4,  0,
                                                                        7,  0, 10, 0, 2,  0, 5,  0, 8,  0, 11, 0,
                                                                        12, 0, 15, 0, 18, 0, 21, 0, 13, 0, 16, 0,
                                                                        19, 0, 22, 0, 14, 0, 17, 0, 20, 0, 23, 0}),
            reference_tests::Tensor({4}, element::i32, std::vector<int32_t>{1, 2, 1, 2}),
            reference_tests::Tensor({2, 3, 4, 2}, IN_ET, std::vector<T>{0,  0, 4,  0, 6,  0, 10, 0, 1,  0, 3,  0,
                                                                        7,  0, 9,  0, 2,  0, 5,  0, 8,  0, 11, 0,
                                                                        12, 0, 16, 0, 18, 0, 22, 0, 13, 0, 15, 0,
                                                                        19, 0, 21, 0, 14, 0, 17, 0, 20, 0, 23, 0})),
        ReverseSequenceParams(
            -2,
            -3,
            reference_tests::Tensor({2, 3, 4, 2}, IN_ET, std::vector<T>{0,  0, 3,  0, 6,  0, 9,  0, 1,  0, 4,  0,
                                                                        7,  0, 10, 0, 2,  0, 5,  0, 8,  0, 11, 0,
                                                                        12, 0, 15, 0, 18, 0, 21, 0, 13, 0, 16, 0,
                                                                        19, 0, 22, 0, 14, 0, 17, 0, 20, 0, 23, 0}),
            reference_tests::Tensor({4}, element::i32, std::vector<int32_t>{1, 2, 1, 2}),
            reference_tests::Tensor({2, 3, 4, 2}, IN_ET, std::vector<T>{0,  0, 4,  0, 6,  0, 10, 0, 1,  0, 3,  0,
                                                                        7,  0, 9,  0, 2,  0, 5,  0, 8,  0, 11, 0,
                                                                        12, 0, 16, 0, 18, 0, 22, 0, 13, 0, 15, 0,
                                                                        19, 0, 21, 0, 14, 0, 17, 0, 20, 0, 23, 0})),
        ReverseSequenceParams(
            0,
            1,
            reference_tests::Tensor({4, 3, 2, 2},
                                    IN_ET,
                                    std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                                   16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                                   32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}),
            reference_tests::Tensor({4}, element::i32, std::vector<int32_t>{1, 2, 3, 3}),
            reference_tests::Tensor({4, 3, 2, 2},
                                    IN_ET,
                                    std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 17, 18, 19,
                                                   12, 13, 14, 15, 20, 21, 22, 23, 32, 33, 34, 35, 28, 29, 30, 31,
                                                   24, 25, 26, 27, 44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39})),
        // 5D
        ReverseSequenceParams(
            0,
            2,
            reference_tests::Tensor(
                {4, 2, 3, 2, 2},
                IN_ET,
                std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                               60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                               80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95}),
            reference_tests::Tensor({4}, element::i32, std::vector<int32_t>{1, 2, 1, 2}),
            reference_tests::Tensor(
                {4, 2, 3, 2, 2},
                IN_ET,
                std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43,
                               36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                               60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 72, 73, 74, 75,
                               80, 81, 82, 83, 88, 89, 90, 91, 84, 85, 86, 87, 92, 93, 94, 95}))};
    return reverseSeqParams;
}

std::vector<ReverseSequenceParams> generateReverseSeqCombinedParams() {
    const std::vector<std::vector<ReverseSequenceParams>> reverseSeqTypeParams{
        generateReverseSeqParams<element::Type_t::f32>(),
        generateReverseSeqParams<element::Type_t::f16>(),
        generateReverseSeqParams<element::Type_t::i32>(),
        generateReverseSeqParams<element::Type_t::u16>(),
        generateReverseSeqParams<element::Type_t::i8>(),
        generateReverseSeqParams<element::Type_t::u8>()};
    std::vector<ReverseSequenceParams> combinedParams;

    for (const auto& params : reverseSeqTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequence_With_Hardcoded_Refs,
                         ReferenceReverseSequenceTest,
                         testing::ValuesIn(generateReverseSeqCombinedParams()),
                         ReferenceReverseSequenceTest::getTestCaseName);
}  // namespace
