// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset1.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GatherTreeParams {
    Tensor stepIds;
    Tensor parentIdx;
    Tensor maxSeqLen;
    Tensor endToken;
    Tensor finalIdx;
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
        if (param.testcaseName != "") {
            result << "_fShape=" << param.finalIdx.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_fShape=" << param.finalIdx.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const GatherTreeParams& params) {
        const auto step_ids = std::make_shared<opset1::Parameter>(params.stepIds.type, params.stepIds.shape);
        const auto parent_idx = std::make_shared<opset1::Parameter>(params.parentIdx.type, params.parentIdx.shape);
        const auto max_seq_len = std::make_shared<opset1::Parameter>(params.maxSeqLen.type, params.maxSeqLen.shape);
        const auto end_token = std::make_shared<opset1::Parameter>(params.endToken.type, params.endToken.shape);
        const auto gather_tree = std::make_shared<opset1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
        const auto f = std::make_shared<Function>(gather_tree, ParameterVector{step_ids, parent_idx, max_seq_len, end_token});
        return f;
    }
};

TEST_P(ReferenceGatherTreeTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<GatherTreeParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<GatherTreeParams> params {
        Builder {}
        .stepIds(Tensor(ET, {1, 1, 10}, std::vector<T>{
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9}))
        .parentIdx(Tensor(ET, {1, 1, 10}, std::vector<T>{
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9}))
        .maxSeqLen(Tensor(ET, {1}, std::vector<T>{7}))
        .endToken(Tensor(ET, {}, std::vector<T>{7}))
        .finalIdx(Tensor(ET, {1, 1, 10}, std::vector<T>{
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9}))
        .testcaseName("gather_tree_dummy")
    };
    return params;
}


std::vector<GatherTreeParams> generateCombinedParams() {
    const std::vector<std::vector<GatherTreeParams>> generatedParams {
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::f32>(),
    };
    std::vector<GatherTreeParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GatherTree_With_Hardcoded_Refs, ReferenceGatherTreeTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceGatherTreeTest::getTestCaseName);
} // namespace