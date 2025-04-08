// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_nd_update.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;
using Reduction = ov::op::v15::ScatterNDUpdate::Reduction;

namespace {
struct ScatterNDUpdateParams {
    ScatterNDUpdateParams(const reference_tests::Tensor& dataTensor,
                          const reference_tests::Tensor& indexTensor,
                          const reference_tests::Tensor& updateTensor,
                          const reference_tests::Tensor& expectedTensor,
                          const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          indexTensor(indexTensor),
          updateTensor(updateTensor),
          expectedTensor(expectedTensor),
          reduction(Reduction::NONE),
          testcaseName(testcaseName) {}

    ScatterNDUpdateParams(const reference_tests::Tensor& dataTensor,
                          const reference_tests::Tensor& indexTensor,
                          const reference_tests::Tensor& updateTensor,
                          const reference_tests::Tensor& expectedTensor,
                          const Reduction paramReduction,
                          const std::string& testcaseName)
        : dataTensor(dataTensor),
          indexTensor(indexTensor),
          updateTensor(updateTensor),
          expectedTensor(expectedTensor),
          reduction{paramReduction},
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor indexTensor;
    reference_tests::Tensor updateTensor;
    reference_tests::Tensor expectedTensor;
    Reduction reduction;
    std::string testcaseName;
};

class ReferenceScatterNDUpdateV3LayerTest : public testing::TestWithParam<ScatterNDUpdateParams>,
                                            public CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

public:
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

class ReferenceScatterNDUpdateV15LayerTest : public testing::TestWithParam<ScatterNDUpdateParams>,
                                             public CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScatterNDUpdateParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << ReferenceScatterNDUpdateV3LayerTest::getTestCaseName(obj);
        result << "_reduction=" << param.reduction;
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
        const auto scatter = std::make_shared<op::v15::ScatterNDUpdate>(data, indices, updates, params.reduction);
        return std::make_shared<ov::Model>(NodeVector{scatter}, ParameterVector{data});
    }
};

TEST_P(ReferenceScatterNDUpdateV3LayerTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceScatterNDUpdateV15LayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET, element::Type_t IU_ET>
std::vector<ScatterNDUpdateParams> generateScatterNDUpdateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<IU_ET>::value_type;
    std::vector<ScatterNDUpdateParams> scatterParams{
        // scatter_nd_update_1x1
        ScatterNDUpdateParams(reference_tests::Tensor({1}, IN_ET, std::vector<T>{1}),
                              reference_tests::Tensor({1}, IU_ET, std::vector<U>{0}),
                              reference_tests::Tensor({}, IN_ET, std::vector<T>{40}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{40}),
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
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor(
                {2, 3, 3},
                IN_ET,
                std::vector<T>{91, 92, 93, 94, 95, 96, 97, 98, 99, 81, 82, 83, 84, 85, 86, 87, 88, 89}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{91, 92, 93, 94, 95, 96, 97, 98, 99,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     81, 82, 83, 84, 85, 86, 87, 88, 89}),
            "scatter_nd_update_3x3_by_1"),
        // scatter_nd_update_3x3_by_2v2
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({2, 2, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 2}),
            reference_tests::Tensor({2, 2}, IN_ET, std::vector<T>{91, 92, 81, 82}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{91, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     81, 22, 23, 24, 25, 26, 27, 28, 82,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 92}),
            "scatter_nd_update_3x3_by_2v2"),
        // scatter_nd_update_3x3_by_2
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 0, 2, 2}),
            reference_tests::Tensor({2, 3}, IN_ET, std::vector<T>{91, 92, 93, 87, 88, 89}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{91, 92, 93, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 87, 88, 89}),
            "scatter_nd_update_3x3_by_2"),
        // scatter_nd_update_3x3_by_3
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
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
        ScatterNDUpdateParams(
            reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
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
        ScatterNDUpdateParams(
            reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                     1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                     8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                                                     8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8}),
            reference_tests::Tensor({2, 2, 2}, IU_ET, std::vector<U>{0, 0, 2, 2, 1, 1, 3, 3}),
            reference_tests::Tensor({2, 2, 4},
                                    IN_ET,
                                    std::vector<T>{15, 16, 17, 18, 25, 26, 27, 28, 35, 36, 37, 38, 45, 46, 47, 58}),
            reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{15, 16, 17, 18, 5,  6, 7, 8,  8,  7,  6,  5, 4,
                                                                     3,  2,  1,  1,  2,  3, 4, 35, 36, 37, 38, 8, 7,
                                                                     6,  5,  4,  3,  2,  1, 8, 7,  6,  5,  4,  3, 2,
                                                                     1,  25, 26, 27, 28, 5, 6, 7,  8,  8,  7,  6, 5,
                                                                     4,  3,  2,  1,  1,  2, 3, 4,  45, 46, 47, 58}),
            "scatter_nd_update_4x4_v2"),
    };
    return scatterParams;
}

template <element::Type_t IN_ET, element::Type_t IU_ET>
std::vector<ScatterNDUpdateParams> generateScatterNDUpdateV15Params() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<IU_ET>::value_type;
    std::vector<ScatterNDUpdateParams> scatterParams{
        // Duplicated indices tests:
        ScatterNDUpdateParams(reference_tests::Tensor({1}, IN_ET, std::vector<T>{1}),
                              reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 0}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{40, 50}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{50}),
                              "scatter_nd_update_1x1_duplicated_indices"),
        ScatterNDUpdateParams(reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 1, 1, 1, 1, 1}),
                              reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 1, 0, 1}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{5, 10}),
                              reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 10, 1, 1, 1, 1}),
                              "scatter_nd_update_tf_example_duplicated_indices"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{8,  12, 1,  5,  3,  19, 2,  4,  13,
                                                                     7,  6,  25, 5,  12, 11, 23, 17, 2,
                                                                     18, 23, 16, 17, 11, 22, 0,  10, 20}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{2, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{4, 15, 11, 13, 15, 23, 1, 4, 4, 11, 2, 10, 1, 25, 25, 19, 23, 10}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{8,  12, 1,  5, 3,  19, 2,  4,  13,
                                                                     7,  6,  25, 5, 12, 11, 23, 17, 2,
                                                                     11, 2,  10, 1, 25, 25, 19, 23, 10}),
            "scatter_nd_update_3x3_by_1_duplicated_indices"),

        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({3, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, -1, 2, -1, 2, -1}),
            reference_tests::Tensor({3}, IN_ET, std::vector<T>{91, 99, 100}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{91, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 100}),
            "scatter_nd_update_3x3_by_3_duplicated_indices"),
        ScatterNDUpdateParams(reference_tests::Tensor({8}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8}),
                              reference_tests::Tensor({5, 1}, IU_ET, std::vector<U>{4, 3, 1, 7, 1}),
                              reference_tests::Tensor({5}, IN_ET, std::vector<T>{9, 10, 11, 12, 22}),
                              reference_tests::Tensor({8}, IN_ET, std::vector<T>{1, 22, 3, 10, 9, 6, 7, 12}),
                              "scatter_nd_update_1d_from_examples_duplicated_indices"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                     1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                     8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                                                     8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8}),
            reference_tests::Tensor({3, 1}, IU_ET, std::vector<U>{0, 2, -2}),
            reference_tests::Tensor({3, 4, 4}, IN_ET, std::vector<T>{5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,
                                                                     8,  8,  8,  8,  1,  1,  1,  1,  2,  2,  2,  2,
                                                                     3,  3,  3,  3,  4,  4,  4,  4,  10, 10, 10, 10,
                                                                     11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13}),
            reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,
                                                                     8,  8,  8,  1,  2,  3,  4,  5,  6,  7,  8,  8,  7,
                                                                     6,  5,  4,  3,  2,  1,  10, 10, 10, 10, 11, 11, 11,
                                                                     11, 12, 12, 12, 12, 13, 13, 13, 13, 8,  7,  6,  5,
                                                                     4,  3,  2,  1,  1,  2,  3,  4,  5,  6,  7,  8}),
            "scatter_nd_update_4x4_shape_from_examples_duplicated_indices"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                     1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
                                                                     8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
                                                                     8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8}),
            reference_tests::Tensor({3, 2, 2}, IU_ET, std::vector<U>{0, 0, 2, 2, 1, 1, 3, 3, 0, 0, 2, 2}),
            reference_tests::Tensor({3, 2, 4}, IN_ET, std::vector<T>{15, 16, 17, 18, 25, 26, 27, 28, 35, 36, 37, 38,
                                                                     45, 46, 47, 48, 55, 56, 57, 58, 65, 66, 67, 68}),
            reference_tests::Tensor({4, 4, 4}, IN_ET, std::vector<T>{55, 56, 57, 58, 5,  6, 7, 8,  8,  7,  6,  5, 4,
                                                                     3,  2,  1,  1,  2,  3, 4, 35, 36, 37, 38, 8, 7,
                                                                     6,  5,  4,  3,  2,  1, 8, 7,  6,  5,  4,  3, 2,
                                                                     1,  65, 66, 67, 68, 5, 6, 7,  8,  8,  7,  6, 5,
                                                                     4,  3,  2,  1,  1,  2, 3, 4,  45, 46, 47, 48}),
            "scatter_nd_update_4x4_v2_duplicated_indices"),
        // Reduction tests:
        // scatter_nd_update_1x1
        ScatterNDUpdateParams(reference_tests::Tensor({1}, IN_ET, std::vector<T>{1}),
                              reference_tests::Tensor({1}, IU_ET, std::vector<U>{0}),
                              reference_tests::Tensor({}, IN_ET, std::vector<T>{40}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{41}),
                              Reduction::SUM,
                              "scatter_nd_update_1x1_SUM"),
        ScatterNDUpdateParams(reference_tests::Tensor({1}, IN_ET, std::vector<T>{40}),
                              reference_tests::Tensor({1}, IU_ET, std::vector<U>{0}),
                              reference_tests::Tensor({}, IN_ET, std::vector<T>{40}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{0}),
                              Reduction::SUB,
                              "scatter_nd_update_1x1_SUB"),
        ScatterNDUpdateParams(reference_tests::Tensor({1}, IN_ET, std::vector<T>{2}),
                              reference_tests::Tensor({1}, IU_ET, std::vector<U>{0}),
                              reference_tests::Tensor({}, IN_ET, std::vector<T>{40}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{80}),
                              Reduction::PROD,
                              "scatter_nd_update_1x1_PROD"),
        ScatterNDUpdateParams(reference_tests::Tensor({1}, IN_ET, std::vector<T>{2}),
                              reference_tests::Tensor({1}, IU_ET, std::vector<U>{0}),
                              reference_tests::Tensor({}, IN_ET, std::vector<T>{40}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{40}),
                              Reduction::MAX,
                              "scatter_nd_update_1x1_MAX"),
        ScatterNDUpdateParams(reference_tests::Tensor({1}, IN_ET, std::vector<T>{2}),
                              reference_tests::Tensor({1}, IU_ET, std::vector<U>{0}),
                              reference_tests::Tensor({}, IN_ET, std::vector<T>{40}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{2}),
                              Reduction::MIN,
                              "scatter_nd_update_1x1_MIN"),
        // scatter_nd_update_tf_example
        ScatterNDUpdateParams(reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 1, 1, 1, 1, 1}),
                              reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 1, 2, 0}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{5, 10}),
                              reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 6, 1, 1, 11, 1}),
                              Reduction::SUM,
                              "scatter_nd_update_tf_example"),
        ScatterNDUpdateParams(reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 6, 1, 1, 11, 1}),
                              reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 1, 2, 0}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{5, 10}),
                              reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 1, 1, 1, 1, 1}),
                              Reduction::SUB,
                              "scatter_nd_update_tf_example"),
        ScatterNDUpdateParams(reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 2, 1, 1, 2, 1}),
                              reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 1, 2, 0}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{5, 10}),
                              reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 10, 1, 1, 20, 1}),
                              Reduction::PROD,
                              "scatter_nd_update_tf_example"),
        ScatterNDUpdateParams(reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 1, 1, 1, 1, 1}),
                              reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 1, 2, 0}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{5, 10}),
                              reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{1, 5, 1, 1, 10, 1}),
                              Reduction::MAX,
                              "scatter_nd_update_tf_example"),
        ScatterNDUpdateParams(reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{15, 15, 15, 15, 15, 15}),
                              reference_tests::Tensor({2, 2}, IU_ET, std::vector<U>{0, 1, 2, 0}),
                              reference_tests::Tensor({2}, IN_ET, std::vector<T>{5, 10}),
                              reference_tests::Tensor({3, 2}, IN_ET, std::vector<T>{15, 5, 15, 15, 10, 15}),
                              Reduction::MIN,
                              "scatter_nd_update_tf_example"),
        // scatter_nd_update_3x3_by_1
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{8,  12, 1,  5,  3,  19, 2,  4,  13,
                                                                     7,  6,  25, 5,  12, 11, 23, 17, 2,
                                                                     18, 23, 16, 17, 11, 22, 0,  10, 20}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{4, 15, 11, 13, 15, 23, 1, 4, 4, 11, 2, 10, 1, 25, 25, 19, 23, 10}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{12, 27, 12, 18, 18, 42, 3,  8,  17,
                                                                     7,  6,  25, 5,  12, 11, 23, 17, 2,
                                                                     29, 25, 26, 18, 36, 47, 19, 33, 30}),
            Reduction::SUM,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{8,  12, 1,  5,  3,  19, 2,  4,  13,
                                                                     7,  6,  25, 5,  12, 11, 23, 17, 2,
                                                                     18, 23, 16, 17, 11, 22, 0,  10, 20}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{4, 15, 11, 13, 15, 23, 1, 4, 4, 11, 2, 10, 1, 25, 25, 19, 23, 10}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{32,  180, 11,  65, 45,  437, 2,  16,  52,
                                                                     7,   6,   25,  5,  12,  11,  23, 17,  2,
                                                                     198, 46,  160, 17, 275, 550, 0,  230, 200}),
            Reduction::PROD,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{8,  12, 1,  5,  3,  19, 2,  4,  13,
                                                                     7,  6,  25, 5,  12, 11, 23, 17, 2,
                                                                     18, 23, 16, 17, 11, 22, 0,  10, 20}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{4, 15, 11, 13, 15, 23, 1, 4, 4, 11, 2, 10, 1, 25, 25, 19, 23, 10}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{8,  15, 11, 13, 15, 23, 2,  4,  13,
                                                                     7,  6,  25, 5,  12, 11, 23, 17, 2,
                                                                     18, 23, 16, 17, 25, 25, 19, 23, 20}),
            Reduction::MAX,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{8,  12, 1,  5,  3,  19, 2,  4,  13,
                                                                     7,  6,  25, 5,  12, 11, 23, 17, 2,
                                                                     18, 23, 16, 17, 11, 22, 0,  10, 20}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{4, 15, 11, 13, 15, 23, 1, 4, 4, 11, 2, 10, 1, 25, 25, 19, 23, 10}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{4,  12, 1,  5, 3,  19, 1,  4, 4,  7,  6, 25, 5, 12,
                                                                     11, 23, 17, 2, 11, 2,  10, 1, 11, 22, 0, 10, 10}),
            Reduction::MIN,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{37, 29, 48, 48, 27, 29, 26, 42, 42,
                                                                     48, 49, 25, 39, 36, 47, 34, 49, 42,
                                                                     35, 37, 49, 42, 46, 26, 41, 31, 41}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor(
                {2, 3, 3},
                IN_ET,
                std::vector<T>{18, 13, 13, 19, 13, 1, 25, 24, 15, 6, 12, 17, 12, 10, 8, 18, 16, 19}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{19, 16, 35, 29, 14, 28, 1,  18, 27,
                                                                     48, 49, 25, 39, 36, 47, 34, 49, 42,
                                                                     29, 25, 32, 30, 36, 18, 23, 15, 22}),
            Reduction::SUB,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({3, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, -1, 2, -1, 2, -1}),
            reference_tests::Tensor({3}, IN_ET, std::vector<T>{2, 4, 6}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{13, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 49}),
            Reduction::SUM,
            "scatter_nd_update_3x3_by_3_duplicated_indices"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({3, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, -1, 2, -1, 2, -1}),
            reference_tests::Tensor({3}, IN_ET, std::vector<T>{2, 4, 6}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{9,  12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 29}),
            Reduction::SUB,
            "scatter_nd_update_3x3_by_3_duplicated_indices"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({3, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, -1, 2, -1, 2, -1}),
            reference_tests::Tensor({3}, IN_ET, std::vector<T>{2, 4, 6}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{22, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 936}),
            Reduction::PROD,
            "scatter_nd_update_3x3_by_3_duplicated_indices"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({3, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, -1, 2, -1, 2, -1}),
            reference_tests::Tensor({3}, IN_ET, std::vector<T>{2, 4, 6}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            Reduction::MAX,
            "scatter_nd_update_3x3_by_3_duplicated_indices"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 39}),
            reference_tests::Tensor({3, 3}, IU_ET, std::vector<U>{0, 0, 0, 2, -1, 2, -1, 2, -1}),
            reference_tests::Tensor({3}, IN_ET, std::vector<T>{2, 4, 6}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{2,  12, 13, 14, 15, 16, 17, 18, 19,
                                                                     21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                                     31, 32, 33, 34, 35, 36, 37, 38, 4}),
            Reduction::MIN,
            "scatter_nd_update_3x3_by_3_duplicated_indices"),
    };
    return scatterParams;
}

template <element::Type_t IU_ET>
std::vector<ScatterNDUpdateParams> generateScatterNDUpdateV15ParamsReductionsBoolean() {
    const auto IN_ET = element::Type_t::boolean;
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<IU_ET>::value_type;
    std::vector<ScatterNDUpdateParams> scatterParams{
        // scatter_nd_update_3x3_by_1
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0}),
            Reduction::SUM,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0}),
            Reduction::PROD,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0}),
            Reduction::MAX,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0}),
            Reduction::MIN,
            "scatter_nd_update_3x3_by_1"),
        ScatterNDUpdateParams(
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0}),
            reference_tests::Tensor({2, 1}, IU_ET, std::vector<U>{0, 2}),
            reference_tests::Tensor({2, 3, 3},
                                    IN_ET,
                                    std::vector<T>{0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0}),
            reference_tests::Tensor({3, 3, 3}, IN_ET, std::vector<T>{1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                                                                     1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0}),
            Reduction::SUB,
            "scatter_nd_update_3x3_by_1"),
    };
    return scatterParams;
}

std::vector<ScatterNDUpdateParams> generateScatterNDUpdateV3CombinedParams() {
    const std::vector<std::vector<ScatterNDUpdateParams>> scatterTypeParams{
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

std::vector<ScatterNDUpdateParams> generateScatterNDUpdateV15CombinedParams() {
    const std::vector<std::vector<ScatterNDUpdateParams>> scatterTypeParams{
        generateScatterNDUpdateV15Params<element::Type_t::i32, element::Type_t::i32>(),
        generateScatterNDUpdateV15Params<element::Type_t::i64, element::Type_t::i32>(),
        generateScatterNDUpdateV15Params<element::Type_t::u32, element::Type_t::i32>(),
        generateScatterNDUpdateV15Params<element::Type_t::u64, element::Type_t::i32>(),
        generateScatterNDUpdateV15Params<element::Type_t::f16, element::Type_t::i32>(),
        generateScatterNDUpdateV15Params<element::Type_t::f32, element::Type_t::i32>(),
        generateScatterNDUpdateV15Params<element::Type_t::i32, element::Type_t::i64>(),
        generateScatterNDUpdateV15Params<element::Type_t::i64, element::Type_t::i64>(),
        generateScatterNDUpdateV15Params<element::Type_t::u32, element::Type_t::i64>(),
        generateScatterNDUpdateV15Params<element::Type_t::u64, element::Type_t::i64>(),
        generateScatterNDUpdateV15Params<element::Type_t::f16, element::Type_t::i64>(),
        generateScatterNDUpdateV15Params<element::Type_t::f32, element::Type_t::i64>(),
        generateScatterNDUpdateV15ParamsReductionsBoolean<element::Type_t::i32>(),
        generateScatterNDUpdateV15ParamsReductionsBoolean<element::Type_t::i64>(),
    };
    std::vector<ScatterNDUpdateParams> combinedParams = generateScatterNDUpdateV3CombinedParams();

    for (const auto& params : scatterTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate_With_Hardcoded_Refs,
                         ReferenceScatterNDUpdateV3LayerTest,
                         testing::ValuesIn(generateScatterNDUpdateV3CombinedParams()),
                         ReferenceScatterNDUpdateV3LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate_With_Hardcoded_Refs,
                         ReferenceScatterNDUpdateV15LayerTest,
                         testing::ValuesIn(generateScatterNDUpdateV15CombinedParams()),
                         ReferenceScatterNDUpdateV15LayerTest::getTestCaseName);

class ReferenceScatterNDUpdateLayerNegativeTest : public ReferenceScatterNDUpdateV3LayerTest {};

TEST_P(ReferenceScatterNDUpdateLayerNegativeTest, CompareWithRefsNegative) {
    LoadNetwork();
    FillInputs();
    inferRequest = executableNetwork.create_infer_request();
    const auto& functionParams = function->get_parameters();

    for (size_t i = 0; i < functionParams.size(); ++i) {
        inferRequest.set_tensor(executableNetwork.input(i), inputData[i]);
    }

    ASSERT_THROW(inferRequest.infer(), ov::Exception);
}

template <element::Type_t IN_ET, element::Type_t IU_ET>
std::vector<ScatterNDUpdateParams> generateScatterNDUpdateNegativeParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    using U = typename element_type_traits<IU_ET>::value_type;
    std::vector<ScatterNDUpdateParams> scatterParams{
        // scatter_nd_update_2x3_with_out_of_bounds_index
        ScatterNDUpdateParams(reference_tests::Tensor({2, 3}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6}),
                              reference_tests::Tensor({1, 2}, IU_ET, std::vector<U>{1, 3}),
                              reference_tests::Tensor({1}, IN_ET, std::vector<T>{10}),
                              reference_tests::Tensor({2, 2}, IN_ET, std::vector<T>{1, 2, 3, 4, 5, 6}),
                              "scatter_nd_update_2x3_with_out_of_bounds_index"),
    };
    return scatterParams;
}

std::vector<ScatterNDUpdateParams> generateScatterNDUpdateCombinedNegativeParams() {
    const std::vector<std::vector<ScatterNDUpdateParams>> scatterTypeParams{
        generateScatterNDUpdateNegativeParams<element::Type_t::f32, element::Type_t::i64>()};
    std::vector<ScatterNDUpdateParams> combinedParams;

    for (const auto& params : scatterTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate_With_Hardcoded_Refs,
                         ReferenceScatterNDUpdateLayerNegativeTest,
                         testing::ValuesIn(generateScatterNDUpdateCombinedNegativeParams()),
                         ReferenceScatterNDUpdateLayerNegativeTest::getTestCaseName);

}  // namespace
