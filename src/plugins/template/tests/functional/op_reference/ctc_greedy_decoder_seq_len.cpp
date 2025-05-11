// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct CTCGreedyDecoderSeqLenParams {
    CTCGreedyDecoderSeqLenParams(const reference_tests::Tensor& dataTensor,
                                 const reference_tests::Tensor& seqLenTensor,
                                 const reference_tests::Tensor& blankIndexTensor,
                                 int64_t mergeRepeated,
                                 const reference_tests::Tensor& expectedTensor,
                                 const reference_tests::Tensor& expectedTensor2,
                                 const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          seqLenTensor(seqLenTensor),
          blankIndexTensor(blankIndexTensor),
          mergeRepeated(mergeRepeated),
          expectedTensor(expectedTensor),
          expectedTensor2(expectedTensor2),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor seqLenTensor;
    reference_tests::Tensor blankIndexTensor;
    bool mergeRepeated;
    reference_tests::Tensor expectedTensor;
    reference_tests::Tensor expectedTensor2;
    std::string testcaseName;
};

struct CTCGreedyDecoderSeqLenParamsNoOptionalInput {
    CTCGreedyDecoderSeqLenParamsNoOptionalInput(const reference_tests::Tensor& dataTensor,
                                                const reference_tests::Tensor& seqLenTensor,
                                                int64_t mergeRepeated,
                                                const reference_tests::Tensor& expectedTensor,
                                                const reference_tests::Tensor& expectedTensor2,
                                                const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          seqLenTensor(seqLenTensor),
          mergeRepeated(mergeRepeated),
          expectedTensor(expectedTensor),
          expectedTensor2(expectedTensor2),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor seqLenTensor;
    bool mergeRepeated;
    reference_tests::Tensor expectedTensor;
    reference_tests::Tensor expectedTensor2;
    std::string testcaseName;
};

class ReferenceCTCGreedyDecoderSeqLenTest : public testing::TestWithParam<CTCGreedyDecoderSeqLenParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.seqLenTensor.data};
        refOutData = {params.expectedTensor.data, params.expectedTensor2.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CTCGreedyDecoderSeqLenParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_sType=" << param.seqLenTensor.type;
        result << "_sShape=" << param.seqLenTensor.shape;
        result << "_bType=" << param.blankIndexTensor.type;
        result << "_bShape=" << param.blankIndexTensor.shape;
        result << "_mRepeated=" << param.mergeRepeated;
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
    static std::shared_ptr<Model> CreateFunction(const CTCGreedyDecoderSeqLenParams& params) {
        std::shared_ptr<Model> function;
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto seq_len = std::make_shared<op::v0::Parameter>(params.seqLenTensor.type, params.seqLenTensor.shape);
        auto blank_index = std::make_shared<op::v0::Constant>(params.blankIndexTensor.type,
                                                              params.blankIndexTensor.shape,
                                                              params.blankIndexTensor.data.data());
        const auto decoder =
            std::make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, blank_index, params.mergeRepeated);
        function = std::make_shared<ov::Model>(decoder->outputs(), ParameterVector{data, seq_len});
        return function;
    }
};

class ReferenceCTCGreedyDecoderSeqLenTestNoOptionalInput
    : public testing::TestWithParam<CTCGreedyDecoderSeqLenParamsNoOptionalInput>,
      public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.seqLenTensor.data};
        refOutData = {params.expectedTensor.data, params.expectedTensor2.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CTCGreedyDecoderSeqLenParamsNoOptionalInput>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_sType=" << param.seqLenTensor.type;
        result << "_sShape=" << param.seqLenTensor.shape;
        result << "_mRepeated=" << param.mergeRepeated;
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
    static std::shared_ptr<Model> CreateFunction(const CTCGreedyDecoderSeqLenParamsNoOptionalInput& params) {
        std::shared_ptr<Model> function;
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto seq_len = std::make_shared<op::v0::Parameter>(params.seqLenTensor.type, params.seqLenTensor.shape);
        const auto decoder = std::make_shared<op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, params.mergeRepeated);
        function = std::make_shared<ov::Model>(decoder->outputs(), ParameterVector{data, seq_len});
        return function;
    }
};

TEST_P(ReferenceCTCGreedyDecoderSeqLenTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceCTCGreedyDecoderSeqLenTestNoOptionalInput, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<CTCGreedyDecoderSeqLenParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<CTCGreedyDecoderSeqLenParams> params{
        CTCGreedyDecoderSeqLenParams(
            reference_tests::Tensor(ET, {1, 3, 3}, std::vector<T>{0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{2}),
            reference_tests::Tensor(element::i32, {}, std::vector<int32_t>{2}),
            false,
            reference_tests::Tensor(element::i32, {1, 3}, std::vector<int32_t>{1, 0, -1}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{2}),
            "evaluate_ctc_greedy_decoder_seq_len"),
        CTCGreedyDecoderSeqLenParams(
            reference_tests::Tensor(ET, {1, 3, 3}, std::vector<T>{0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{2}),
            reference_tests::Tensor(element::i32, {}, std::vector<int32_t>{2}),
            true,
            reference_tests::Tensor(element::i32, {1, 3}, std::vector<int32_t>{1, 0, -1}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{2}),
            "evaluate_ctc_greedy_decoder_seq_len_merge"),
        CTCGreedyDecoderSeqLenParams(
            reference_tests::Tensor(ET,
                                    {2, 3, 3},
                                    std::vector<T>{0.1f,
                                                   0.2f,
                                                   0.f,
                                                   0.15f,
                                                   0.25f,
                                                   0.f,
                                                   0.4f,
                                                   0.3f,
                                                   0.f,
                                                   0.45f,
                                                   0.35f,
                                                   0.f,
                                                   0.5f,
                                                   0.6f,
                                                   0.f,
                                                   0.55f,
                                                   0.65f,
                                                   0.f}),
            reference_tests::Tensor(element::i32, {2}, std::vector<int32_t>{1, 1}),
            reference_tests::Tensor(element::i32, {}, std::vector<int32_t>{2}),
            false,
            reference_tests::Tensor(element::i32, {2, 3}, std::vector<int32_t>{1, -1, -1, 0, -1, -1}),
            reference_tests::Tensor(element::i32, {2}, std::vector<int32_t>{1, 1}),
            "evaluate_ctc_greedy_decoder_seq_len_multiple_batches"),
        CTCGreedyDecoderSeqLenParams(
            reference_tests::Tensor(ET, {3, 3, 3}, std::vector<T>{0.1f,  0.2f,  0.f,   0.15f, 0.25f, 0.f,  0.4f,
                                                                  0.3f,  0.f,   0.45f, 0.35f, 0.f,   0.5f, 0.6f,
                                                                  0.f,   0.55f, 0.65f, 0.f,   0.1f,  0.2f, 0.f,
                                                                  0.15f, 0.25f, 0.f,   0.4f,  0.3f,  0.f}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{2, 3, 1}),
            reference_tests::Tensor(element::i32, {}, std::vector<int32_t>{2}),
            false,
            reference_tests::Tensor(element::i32, {3, 3}, std::vector<int32_t>{1, 1, -1, 0, 1, 1, 1, -1, -1}),
            reference_tests::Tensor(element::i32, {3}, std::vector<int32_t>{2, 3, 1}),
            "evaluate_ctc_greedy_decoder_seq_len_multiple_batches2"),
    };
    return params;
}

std::vector<CTCGreedyDecoderSeqLenParams> generateCombinedParams() {
    const std::vector<std::vector<CTCGreedyDecoderSeqLenParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<CTCGreedyDecoderSeqLenParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

template <element::Type_t ET>
std::vector<CTCGreedyDecoderSeqLenParamsNoOptionalInput> generateParamsNoOptionalInput() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<CTCGreedyDecoderSeqLenParamsNoOptionalInput> params{
        CTCGreedyDecoderSeqLenParamsNoOptionalInput(
            reference_tests::Tensor(ET, {1, 3, 3}, std::vector<T>{0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{2}),
            false,
            reference_tests::Tensor(element::i32, {1, 3}, std::vector<int32_t>{1, 0, -1}),
            reference_tests::Tensor(element::i32, {1}, std::vector<int32_t>{2}),
            "evaluate_ctc_greedy_decoder_seq_len_no_optional_input"),
    };
    return params;
}

std::vector<CTCGreedyDecoderSeqLenParamsNoOptionalInput> generateCombinedParamsNoOptionalInput() {
    const std::vector<std::vector<CTCGreedyDecoderSeqLenParamsNoOptionalInput>> generatedParams{
        generateParamsNoOptionalInput<element::Type_t::bf16>(),
        generateParamsNoOptionalInput<element::Type_t::f16>(),
        generateParamsNoOptionalInput<element::Type_t::f32>(),
        generateParamsNoOptionalInput<element::Type_t::f64>(),
    };
    std::vector<CTCGreedyDecoderSeqLenParamsNoOptionalInput> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoderSeqLen_With_Hardcoded_Refs,
                         ReferenceCTCGreedyDecoderSeqLenTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceCTCGreedyDecoderSeqLenTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoderSeqLen_With_Hardcoded_Refs,
                         ReferenceCTCGreedyDecoderSeqLenTestNoOptionalInput,
                         testing::ValuesIn(generateCombinedParamsNoOptionalInput()),
                         ReferenceCTCGreedyDecoderSeqLenTestNoOptionalInput::getTestCaseName);
}  // namespace
