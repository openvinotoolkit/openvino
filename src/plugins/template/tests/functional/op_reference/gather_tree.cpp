// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_tree.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GatherTreeParams {
    reference_tests::Tensor stepIds;
    reference_tests::Tensor parentIdx;
    reference_tests::Tensor maxSeqLen;
    reference_tests::Tensor endToken;
    reference_tests::Tensor finalIdx;
    std::string testcaseName;
};

struct Builder : ParamsBuilder<GatherTreeParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, stepIds);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, parentIdx);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, maxSeqLen);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, endToken);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, finalIdx);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, testcaseName);
};

class ReferenceGatherTreeTest : public testing::TestWithParam<GatherTreeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.stepIds.data, params.parentIdx.data, params.maxSeqLen.data, params.endToken.data};
        refOutData = {params.finalIdx.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GatherTreeParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "sType=" << param.stepIds.type;
        result << "_sShape=" << param.stepIds.shape;
        result << "_pType=" << param.parentIdx.type;
        result << "_pShape=" << param.parentIdx.shape;
        result << "_mType=" << param.maxSeqLen.type;
        result << "_mShape=" << param.maxSeqLen.shape;
        result << "_eType=" << param.endToken.type;
        result << "_eShape=" << param.endToken.shape;
        result << "_fType=" << param.finalIdx.type;
        result << "_fShape=" << param.finalIdx.shape;
        if (!param.testcaseName.empty()) {
            result << "_=" << param.testcaseName;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const GatherTreeParams& params) {
        const auto step_ids = std::make_shared<op::v0::Parameter>(params.stepIds.type, params.stepIds.shape);
        const auto parent_idx = std::make_shared<op::v0::Parameter>(params.parentIdx.type, params.parentIdx.shape);
        const auto max_seq_len = std::make_shared<op::v0::Parameter>(params.maxSeqLen.type, params.maxSeqLen.shape);
        const auto end_token = std::make_shared<op::v0::Parameter>(params.endToken.type, params.endToken.shape);
        const auto gather_tree = std::make_shared<op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
        const auto f =
            std::make_shared<Model>(gather_tree, ParameterVector{step_ids, parent_idx, max_seq_len, end_token});
        return f;
    }
};

TEST_P(ReferenceGatherTreeTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<GatherTreeParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<GatherTreeParams> params{
        Builder{}
            .stepIds(reference_tests::Tensor(ET, {1, 1, 10}, std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 9}))
            .parentIdx(reference_tests::Tensor(ET, {1, 1, 10}, std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 9}))
            .maxSeqLen(reference_tests::Tensor(ET, {1}, std::vector<T>{9}))
            .endToken(reference_tests::Tensor(ET, {}, std::vector<T>{9}))
            .finalIdx(reference_tests::Tensor(ET, {1, 1, 10}, std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 9}))
            .testcaseName("gather_tree_1"),

        Builder{}
            .stepIds(reference_tests::Tensor(ET, {5, 1, 10}, std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 2, 3, 1, 4,
                                                                            2, 4, 4, 7, 4, 9, 5, 8, 4, 3, 7, 5, 2,
                                                                            4, 8, 3, 1, 5, 7, 9, 4, 5, 6, 4, 2, 9,
                                                                            2, 8, 8, 7, 9, 8, 3, 1, 7, 5, 9}))
            .parentIdx(reference_tests::Tensor(ET, {5, 1, 10}, std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 2, 3, 1, 4,
                                                                              2, 4, 4, 7, 4, 9, 5, 8, 4, 3, 7, 5, 2,
                                                                              4, 8, 3, 1, 5, 7, 9, 4, 5, 6, 4, 2, 9,
                                                                              2, 8, 8, 7, 9, 8, 3, 1, 7, 5, 9}))
            .maxSeqLen(reference_tests::Tensor(ET, {1}, std::vector<T>{9}))
            .endToken(reference_tests::Tensor(ET, {}, std::vector<T>{9}))
            .finalIdx(reference_tests::Tensor(ET, {5, 1, 10}, std::vector<T>{4, 4, 9, 9, 4, 9, 2, 9, 9, 9, 1, 1, 9,
                                                                             9, 1, 9, 9, 9, 9, 9, 1, 1, 9, 9, 1, 9,
                                                                             9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                                                                             9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}))
            .testcaseName("gather_tree_5"),

        Builder{}
            .stepIds(reference_tests::Tensor(
                ET,
                {20, 1, 10},
                std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 2, 3, 1, 4, 2, 4, 4, 7, 4, 9, 5, 8, 4, 3, 7, 5, 2, 4, 8, 3,
                               1, 5, 7, 9, 4, 5, 6, 4, 2, 9, 2, 8, 8, 7, 9, 8, 3, 1, 7, 5, 8, 8, 9, 8, 1, 8, 1, 3, 2,
                               1, 8, 7, 1, 6, 4, 7, 9, 4, 5, 2, 7, 3, 3, 2, 7, 8, 8, 4, 1, 1, 7, 6, 9, 6, 7, 3, 3, 5,
                               8, 2, 1, 1, 5, 5, 9, 1, 3, 9, 3, 2, 2, 5, 1, 1, 7, 9, 2, 9, 3, 3, 5, 6, 1, 6, 6, 6, 2,
                               9, 6, 3, 7, 3, 1, 5, 4, 9, 7, 5, 4, 5, 1, 7, 5, 1, 6, 2, 5, 8, 9, 1, 6, 8, 9, 5, 2, 5,
                               2, 9, 8, 4, 4, 5, 2, 6, 9, 4, 4, 6, 7, 6, 7, 2, 8, 7, 6, 6, 7, 4, 4, 7, 3, 4, 9, 7, 4,
                               8, 9, 1, 6, 5, 6, 1, 2, 8, 9, 1, 5, 4, 6, 9, 4, 4, 3, 7, 9, 7, 6, 3, 1, 7, 9}))
            .parentIdx(reference_tests::Tensor(
                ET,
                {20, 1, 10},
                std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 2, 3, 1, 4, 2, 4, 4, 7, 4, 9, 5, 8, 4, 3, 7, 5, 2, 4, 8, 3,
                               1, 5, 7, 9, 4, 5, 6, 4, 2, 9, 2, 8, 8, 7, 9, 8, 3, 1, 7, 5, 8, 8, 9, 8, 1, 8, 1, 3, 2,
                               1, 8, 7, 1, 6, 4, 7, 9, 4, 5, 2, 7, 3, 3, 2, 7, 8, 8, 4, 1, 1, 7, 6, 9, 6, 7, 3, 3, 5,
                               8, 2, 1, 1, 5, 5, 9, 1, 3, 9, 3, 2, 2, 5, 1, 1, 7, 9, 2, 9, 3, 3, 5, 6, 1, 6, 6, 6, 2,
                               9, 6, 3, 7, 3, 1, 5, 4, 9, 7, 5, 4, 5, 1, 7, 5, 1, 6, 2, 5, 8, 9, 1, 6, 8, 9, 5, 2, 5,
                               2, 9, 8, 4, 4, 5, 2, 6, 9, 4, 4, 6, 7, 6, 7, 2, 8, 7, 6, 6, 7, 4, 4, 7, 3, 4, 9, 7, 4,
                               8, 9, 1, 6, 5, 6, 1, 2, 8, 9, 1, 5, 4, 6, 9, 4, 4, 3, 7, 9, 7, 6, 3, 1, 7, 9}))
            .maxSeqLen(reference_tests::Tensor(ET, {1}, std::vector<T>{9}))
            .endToken(reference_tests::Tensor(ET, {}, std::vector<T>{9}))
            .finalIdx(reference_tests::Tensor(
                ET,
                {20, 1, 10},
                std::vector<T>{9, 4, 9, 4, 4, 4, 9, 4, 9, 9, 9, 1, 9, 1, 1, 1, 9, 1, 9, 9, 9, 1, 9, 1, 1, 1, 9, 1, 9,
                               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                               9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}))
            .testcaseName("gather_tree_10"),
    };
    return params;
}

std::vector<GatherTreeParams> generateCombinedParams() {
    const std::vector<std::vector<GatherTreeParams>> generatedParams{
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::f32>(),
    };
    std::vector<GatherTreeParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GatherTree_With_Hardcoded_Refs,
                         ReferenceGatherTreeTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceGatherTreeTest::getTestCaseName);
}  // namespace
