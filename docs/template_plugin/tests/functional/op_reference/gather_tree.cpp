// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/gather_tree.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GatherTreeParams {
    template <class IN_ET>
    GatherTreeParams(const ov::Shape inShape, std::vector<IN_ET> stepIds, const std::vector<IN_ET> parentIds,
        const std::vector<IN_ET> maxSeqLen, const std::vector<IN_ET> endToken, std::vector<IN_ET> output) :
        stepIdsTensor(inShape, element::from<IN_ET>(), stepIds), parentIdsTensor(inShape, element::from<IN_ET>(), parentIds),
        maxSeqLenTensor(ov::Shape{inShape[1]}, element::from<IN_ET>(), maxSeqLen), endTokenTensor(ov::Shape{}, element::from<IN_ET>(), endToken),
        expectedTensor(inShape, element::from<IN_ET>(), output) {}
    Tensor stepIdsTensor;
    Tensor parentIdsTensor;
    Tensor maxSeqLenTensor;
    Tensor endTokenTensor;
    Tensor expectedTensor;
};

class ReferenceGatherTreeTest : public testing::TestWithParam<GatherTreeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.stepIdsTensor.data, params.parentIdsTensor.data, params.maxSeqLenTensor.data, params.endTokenTensor.data};
        refOutData = {params.expectedTensor.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GatherTreeParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.stepIdsTensor.type << "_";
        result << "iShape=" << param.stepIdsTensor.shape;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const GatherTreeParams& params) {
        const auto stepIds = std::make_shared<op::v0::Parameter>(params.stepIdsTensor.type, params.stepIdsTensor.shape);
        const auto parentIds = std::make_shared<op::v0::Parameter>(params.parentIdsTensor.type, params.parentIdsTensor.shape);
        const auto maxSeqLen = std::make_shared<op::v0::Parameter>(params.maxSeqLenTensor.type, params.maxSeqLenTensor.shape);
        const auto endToken = std::make_shared<op::v0::Parameter>(params.endTokenTensor.type, params.endTokenTensor.shape);
        const auto gatherTree = std::make_shared<op::v1::GatherTree>(stepIds, parentIds, maxSeqLen, endToken);
        return std::make_shared<ov::Function>(NodeVector {gatherTree}, ParameterVector {stepIds, parentIds, maxSeqLen, endToken});
    }
};

TEST_P(ReferenceGatherTreeTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<GatherTreeParams> generateGatherTreeParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<GatherTreeParams> gatherTreeParams {
        GatherTreeParams(Shape{4, 1, 3},
                         std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1},
                         std::vector<T>{0, 0, 0, 0, 1, 1, 2, 1, 2, -1, -1, -1},
                         std::vector<T>{3},
                         std::vector<T>{10},
                         std::vector<T>{2, 2, 2, 6, 5, 6, 7, 8, 9, 10, 10, 10}),
        GatherTreeParams(Shape{2, 2, 2},
                         std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8},
                         std::vector<T>{0, 0, 0, 0, 0, 0, 0, 0},
                         std::vector<T>{2, 4},
                         std::vector<T>{0},
                         std::vector<T>{1, 1, 3, 3, 5, 6, 7, 8})
    };
    return gatherTreeParams;
}

std::vector<GatherTreeParams> generateGatherTreeCombinedParams() {
    const std::vector<std::vector<GatherTreeParams>> gatherTreeTypeParams {
        generateGatherTreeParams<element::Type_t::f32>(),
        generateGatherTreeParams<element::Type_t::i32>()};
    std::vector<GatherTreeParams> combinedParams;

    for (const auto& params : gatherTreeTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GatherTree_With_Hardcoded_Refs, ReferenceGatherTreeTest,
    testing::ValuesIn(generateGatherTreeCombinedParams()), ReferenceGatherTreeTest::getTestCaseName);
} // namespace