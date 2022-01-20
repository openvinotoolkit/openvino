// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/constant.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ScatterNDUpdateParams {
    ScatterNDUpdateParams(const reference_tests::Tensor& dataTensor, const reference_tests::Tensor& indexTensor, const reference_tests::Tensor& updateTensor,
                const reference_tests::Tensor& expectedTensor, const std::string& testcaseName = "") :
                dataTensor(dataTensor), indexTensor(indexTensor), updateTensor(updateTensor),
                expectedTensor(expectedTensor), testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor indexTensor;
    reference_tests::Tensor updateTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceScatterNDUpdateLayerTest : public testing::TestWithParam<ScatterNDUpdateParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ScatterNDUpdateParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_iType=" << param.indexTensor.type;
        result << "_iShape=" << param.indexTensor.shape;
        result << "_uType=" << param.updateTensor.type;
        result << "_uShape=" << param.updateTensor.shape;
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
    static std::shared_ptr<Model> CreateFunction(const ScatterNDUpdateParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto indices = std::make_shared<op::v0::Constant>(params.indexTensor.type,
                                                                params.indexTensor.shape,
                                                                params.indexTensor.data.data());
        const auto updates = std::make_shared<op::v0::Constant>(params.updateTensor.type,
                                                                params.updateTensor.shape,
                                                                params.updateTensor.data.data());
        const auto scatter = std::make_shared<op::v3::ScatterNDUpdate>(data, indices, updates);
        return std::make_shared<ov::Model>(NodeVector{scatter}, ParameterVector{data});
    }
};

TEST_P(ReferenceScatterNDUpdateLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET, element::Type_t IU_ET>
std::vector<ScatterNDUpdateParams> generateScatterNDUpdateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<IU_ET>::value_type;
    std::vector<ScatterNDUpdateParams> scatterParams {
        // scatter_nd_update_1x1
        ScatterNDUpdateParams(reference_tests::Tensor({1}, IN_ET, std::vector<T>{1}),
                              reference_tests::Tensor({1}, IU_ET, std::vector<U>{0}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{20}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{20}),
                              "scatter_nd_update_1x1"),
        // scatter_nd_update_2x2_by_1
        ScatterNDUpdateParams(reference_tests::Tensor({2, 2}, IN_ET, std::vector<T>{1, 2, 3, 4}),
                              reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{1, 0}),
                              reference_tests::Tensor({2, 2}, IN_ET, std::vector<T>{10, 20, 30, 40}),
                              reference_tests::Tensor({2, 2}, IN_ET, std::vector<T>{30, 40, 10, 20}),
                              "scatter_nd_update_2x2_by_1"),
        // scatter_nd_update_2x2_by_2
        ScatterNDUpdateParams(reference_tests::Tensor({2, 2}, IN_ET, std::vector<T>{1, 2, 3, 4}),
                              reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 0, 1, 1}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{10, 40}),
                              reference_tests::Tensor({2, 2}, IN_ET, std::vector<T>{10, 2, 3, 40}),
                              "scatter_nd_update_2x2_by_2"),
        // scatter_nd_update_3x3_by_1
        ScatterNDUpdateParams(reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                      21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                      31, 32, 33, 34, 35, 36, 37, 38, 39}),
                              reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
                              reference_tests::Tensor({2, 3, 3}, IN_ET, std::vector<T>{91, 92, 93, 94, 95, 96, 97, 98, 99,
                                                                      81, 82, 83, 84, 85, 86, 87, 88, 89}),
                              reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{91, 92, 93, 94, 95, 96, 97, 98, 99,
                                                                      21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                      81, 82, 83, 84, 85, 86, 87, 88, 89}),
                              "scatter_nd_update_3x3_by_1"),
        // scatter_nd_update_3x3_by_2v2
        ScatterNDUpdateParams(reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                      21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                      31, 32, 33, 34, 35, 36, 37, 38, 39}),
                              reference_tests::Tensor({2, 2, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 2}),
                              reference_tests::Tensor({2, 2}, IN_ET, std::vector<T>{91, 92, 81, 82}),
                              reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{91, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                      81, 22, 23, 24, 25, 26, 27, 28, 82,
                                                                      31, 32, 33, 34, 35, 36, 37, 38, 92}),
                              "scatter_nd_update_3x3_by_2v2"),
        // scatter_nd_update_3x3_by_2
        ScatterNDUpdateParams(reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                      21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                      31, 32, 33, 34, 35, 36, 37, 38, 39}),
                              reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 0, 2, 2}),
                              reference_tests::Tensor({2, 3}, IN_ET, std::vector<T>{91, 92, 93, 87, 88, 89}),
                              reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{91, 92, 93, 14, 15, 16, 17, 18, 19,
                                                                      21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                      31, 32, 33, 34, 35, 36, 87, 88, 89}),
                              "scatter_nd_update_3x3_by_2"),
        // scatter_nd_update_3x3_by_3
        ScatterNDUpdateParams(reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                      21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                      31, 32, 33, 34, 35, 36, 37, 38, 39}),
                              reference_tests::Tensor({2, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, 2, 2}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{91, 99}),
                              reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{91, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                      21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                      31, 32, 33, 34, 35, 36, 37, 38, 99}),
                              "scatter_nd_update_3x3_by_3"),
        // scatter_nd_update_1d_from_examples
        ScatterNDUpdateParams(reference_tests::Tensor({8}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8}),
                              reference_tests::Tensor({4, 1}, IU_ET, std::vector<U>{4, 3, 1, 7}),
                              reference_tests::Tensor({4}, IN_ET, std::vector<T>{9, 10, 11, 12}),
                              reference_tests::Tensor({8}, IN_ET, std::vector<T>{1, 11, 3, 10, 9, 6, 7, 12}),
                              "scatter_nd_update_1d_from_examples"),
        // scatter_nd_update_4x4_shape_from_examples
        ScatterNDUpdateParams(reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                      1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                                                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8}),
                              reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
                              reference_tests::Tensor({2, 4, 4}, IN_ET, std::vector<T>{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                                                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4}),
                              reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
                                                                      1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                                                                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8}),
                              "scatter_nd_update_4x4_shape_from_examples"),
        // scatter_nd_update_4x4_v2
        ScatterNDUpdateParams(reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                      1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                                                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8}),
                              reference_tests::Tensor({2, 2, 2}, IU_ET, std::vector<U>{0, 0, 2, 2, 1, 1, 3, 3}),
                              reference_tests::Tensor({2, 2, 4}, IN_ET, std::vector<T>{15, 16, 17, 18, 25, 26, 27, 28,
                                                                      35, 36, 37, 38, 45, 46, 47, 58}),
                              reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{15, 16, 17, 18, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                      1, 2, 3, 4, 35, 36, 37, 38, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                      8, 7, 6, 5, 4, 3, 2, 1, 25, 26, 27, 28, 5, 6, 7, 8,
                                                                      8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 45, 46, 47, 58}),
                              "scatter_nd_update_4x4_v2"),
    };
    return scatterParams;
}

std::vector<ScatterNDUpdateParams> generateScatterNDUpdateCombinedParams() {
    const std::vector<std::vector<ScatterNDUpdateParams>> scatterTypeParams {
        generateScatterNDUpdateParams<element::Type_t::i32, element::Type_t::i32>(),
        generateScatterNDUpdateParams<element::Type_t::i64, element::Type_t::i32>(),
        generateScatterNDUpdateParams<element::Type_t::u32, element::Type_t::i32>(),
        generateScatterNDUpdateParams<element::Type_t::u64, element::Type_t::i32>(),
        generateScatterNDUpdateParams<element::Type_t::f16, element::Type_t::i32>(),
        generateScatterNDUpdateParams<element::Type_t::f32, element::Type_t::i32>(),
        generateScatterNDUpdateParams<element::Type_t::boolean, element::Type_t::i32>(),
        generateScatterNDUpdateParams<element::Type_t::i32, element::Type_t::i64>(),
        generateScatterNDUpdateParams<element::Type_t::i64, element::Type_t::i64>(),
        generateScatterNDUpdateParams<element::Type_t::u32, element::Type_t::i64>(),
        generateScatterNDUpdateParams<element::Type_t::u64, element::Type_t::i64>(),
        generateScatterNDUpdateParams<element::Type_t::f16, element::Type_t::i64>(),
        generateScatterNDUpdateParams<element::Type_t::f32, element::Type_t::i64>(),
        generateScatterNDUpdateParams<element::Type_t::boolean, element::Type_t::i64>(),
    };
    std::vector<ScatterNDUpdateParams> combinedParams;

    for (const auto& params : scatterTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate_With_Hardcoded_Refs, ReferenceScatterNDUpdateLayerTest,
    testing::ValuesIn(generateScatterNDUpdateCombinedParams()), ReferenceScatterNDUpdateLayerTest::getTestCaseName);
} // namespace
