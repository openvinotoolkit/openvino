// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_greedy_decoder.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct CTCGreedyDecoderParams {
    CTCGreedyDecoderParams(const reference_tests::Tensor& dataTensor,
                           const reference_tests::Tensor& masksTensor,
                           int64_t ctcMergedRepeat,
                           const reference_tests::Tensor& expectedTensor,
                           const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          masksTensor(masksTensor),
          ctcMergedRepeat(ctcMergedRepeat),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor masksTensor;
    bool ctcMergedRepeat;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceCTCGreedyDecoderTest : public testing::TestWithParam<CTCGreedyDecoderParams>,
                                      public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.masksTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CTCGreedyDecoderParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_aType=" << param.masksTensor.type;
        result << "_aShape=" << param.masksTensor.shape;
        result << "_bDims=" << param.ctcMergedRepeat;
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
    static std::shared_ptr<Model> CreateFunction(const CTCGreedyDecoderParams& params) {
        std::shared_ptr<Model> function;
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto indices = std::make_shared<op::v0::Parameter>(params.masksTensor.type, params.masksTensor.shape);
        const auto decoder = std::make_shared<op::v0::CTCGreedyDecoder>(data, indices, params.ctcMergedRepeat);
        function = std::make_shared<ov::Model>(NodeVector{decoder}, ParameterVector{data, indices});
        return function;
    }
};

TEST_P(ReferenceCTCGreedyDecoderTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<CTCGreedyDecoderParams> generateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<CTCGreedyDecoderParams> params{
        CTCGreedyDecoderParams(
            reference_tests::Tensor(IN_ET,
                                    {3, 1, 3},
                                    std::vector<T>{0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f}),
            reference_tests::Tensor(IN_ET, {3, 1}, std::vector<T>{1.0f, 1.0f, 1.0f}),
            false,
            reference_tests::Tensor(IN_ET, {1, 3, 1, 1}, std::vector<T>{1.0f, 0.0f, 1.0f}),
            "ctc_greedy_decoder"),
        CTCGreedyDecoderParams(
            reference_tests::Tensor(IN_ET,
                                    {3, 2, 3},
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
            reference_tests::Tensor(IN_ET, {3, 2}, std::vector<T>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}),
            false,
            reference_tests::Tensor(IN_ET, {2, 3, 1, 1}, std::vector<T>{1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}),
            "ctc_greedy_decoder_multiple_batches"),
        CTCGreedyDecoderParams(
            reference_tests::Tensor(IN_ET,
                                    {3, 1, 3},
                                    std::vector<T>{0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f}),
            reference_tests::Tensor(IN_ET, {3, 1}, std::vector<T>{1.0f, 1.0f, 0.0f}),
            false,
            reference_tests::Tensor(IN_ET, {1, 3, 1, 1}, std::vector<T>{1.0f, 0.0f, -1.0f}),
            "ctc_greedy_decoder_single_batch_short_sequence"),
        CTCGreedyDecoderParams(
            reference_tests::Tensor(IN_ET,
                                    {3, 1, 3},
                                    std::vector<T>{0.1f, 0.2f, 0.f, 0.3f, 0.4f, 0.f, 0.6f, 0.5f, 0.f}),
            reference_tests::Tensor(IN_ET, {3, 1}, std::vector<T>{1.0f, 1.0f, 1.0f}),
            true,
            reference_tests::Tensor(IN_ET, {1, 3, 1, 1}, std::vector<T>{1.0f, 0.0f, -1.0f}),
            "ctc_greedy_decoder_merge"),
        CTCGreedyDecoderParams(
            reference_tests::Tensor(IN_ET,
                                    {3, 1, 3},
                                    std::vector<T>{0.1f, 0.2f, 0.f, 0.3f, 0.4f, 0.f, 0.6f, 0.5f, 0.f}),
            reference_tests::Tensor(IN_ET, {3, 1}, std::vector<T>{1.0f, 1.0f, 1.0f}),
            false,
            reference_tests::Tensor(IN_ET, {1, 3, 1, 1}, std::vector<T>{1.0f, 1.0f, 0.0f}),
            "ctc_greedy_decoder_single_no_merge"),
        CTCGreedyDecoderParams(
            reference_tests::Tensor(IN_ET,
                                    {2, 2, 3},
                                    std::vector<T>{0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f, 0.7f, 0.8f, 0.f}),
            reference_tests::Tensor(IN_ET, {2, 2}, std::vector<T>{1.0f, 1.0f, 1.0f, 0.0f}),
            false,
            reference_tests::Tensor(IN_ET, {2, 2, 1, 1}, std::vector<T>{1.0f, 1.0f, 0.0f, -1.0f}),
            "ctc_greedy_decoder_multiple_sequences"),
    };
    return params;
}

std::vector<CTCGreedyDecoderParams> generateCombinedParams() {
    const std::vector<std::vector<CTCGreedyDecoderParams>> generatedParams{
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<CTCGreedyDecoderParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoder_With_Hardcoded_Refs,
                         ReferenceCTCGreedyDecoderTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceCTCGreedyDecoderTest::getTestCaseName);
}  // namespace
