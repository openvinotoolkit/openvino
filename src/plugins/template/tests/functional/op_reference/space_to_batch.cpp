// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_batch.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct SpaceToBatchParams {
    SpaceToBatchParams(const reference_tests::Tensor& dataTensor,
                       const reference_tests::Tensor& blockShapeTensor,
                       const reference_tests::Tensor& padsBeginTensor,
                       const reference_tests::Tensor& padsEndTensor,
                       const reference_tests::Tensor& expectedTensor,
                       const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          blockShapeTensor(blockShapeTensor),
          padsBeginTensor(padsBeginTensor),
          padsEndTensor(padsEndTensor),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor blockShapeTensor;
    reference_tests::Tensor padsBeginTensor;
    reference_tests::Tensor padsEndTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceSpaceToBatchLayerTest : public testing::TestWithParam<SpaceToBatchParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data,
                     params.blockShapeTensor.data,
                     params.padsBeginTensor.data,
                     params.padsEndTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SpaceToBatchParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_bsType=" << param.blockShapeTensor.type;
        result << "_bsShape=" << param.blockShapeTensor.shape;
        result << "_pbType=" << param.padsBeginTensor.type;
        result << "_pbShape=" << param.padsBeginTensor.shape;
        result << "_peType=" << param.padsEndTensor.type;
        result << "_peShape=" << param.padsEndTensor.shape;
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
    static std::shared_ptr<Model> CreateFunction(const SpaceToBatchParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto blockShape = std::make_shared<op::v0::Parameter>(element::i64, params.blockShapeTensor.shape);
        const auto padsBegin = std::make_shared<op::v0::Parameter>(element::i64, params.padsBeginTensor.shape);
        const auto padsEnd = std::make_shared<op::v0::Parameter>(element::i64, params.padsEndTensor.shape);
        const auto batchToSpace = std::make_shared<op::v1::SpaceToBatch>(data, blockShape, padsBegin, padsEnd);
        return std::make_shared<ov::Model>(NodeVector{batchToSpace},
                                           ParameterVector{data, blockShape, padsBegin, padsEnd});
    }
};

TEST_P(ReferenceSpaceToBatchLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SpaceToBatchParams> generateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<SpaceToBatchParams> batchToSpaceParams{
        // space_to_batch_3D
        SpaceToBatchParams(reference_tests::Tensor({1, 2, 2}, IN_ET, std::vector<T>{1, 1, 1, 1}),
                           reference_tests::Tensor({3}, element::i64, std::vector<int64_t>{1, 1, 1}),
                           reference_tests::Tensor({3}, element::i64, std::vector<int64_t>{0, 0, 0}),
                           reference_tests::Tensor({3}, element::i64, std::vector<int64_t>{0, 0, 0}),
                           reference_tests::Tensor({1, 2, 2}, IN_ET, std::vector<T>{1, 1, 1, 1}),
                           "space_to_batch_4D"),

        // space_to_batch_4D
        SpaceToBatchParams(reference_tests::Tensor({1, 1, 2, 2}, IN_ET, std::vector<T>{1, 1, 1, 1}),
                           reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 1, 1}),
                           reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
                           reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
                           reference_tests::Tensor({1, 1, 2, 2}, IN_ET, std::vector<T>{1, 1, 1, 1}),
                           "space_to_batch_4D"),

        // space_to_batch_5D
        SpaceToBatchParams(reference_tests::Tensor({1, 1, 3, 2, 1}, IN_ET, std::vector<T>{1, 1, 1, 1, 1, 1}),
                           reference_tests::Tensor({5}, element::i64, std::vector<int64_t>{1, 1, 3, 2, 2}),
                           reference_tests::Tensor({5}, element::i64, std::vector<int64_t>{0, 0, 1, 0, 3}),
                           reference_tests::Tensor({5}, element::i64, std::vector<int64_t>{0, 0, 2, 0, 0}),
                           reference_tests::Tensor(
                               {12, 1, 2, 1, 2},
                               IN_ET,
                               std::vector<T>{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                                              0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}),
                           "space_to_batch_5D"),

        // space_to_batch_4x4
        SpaceToBatchParams(reference_tests::Tensor({1, 1, 4, 4},
                                                   IN_ET,
                                                   std::vector<T>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
                           reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{1, 1, 1, 1}),
                           reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 1, 0}),
                           reference_tests::Tensor({4}, element::i64, std::vector<int64_t>{0, 0, 0, 0}),
                           reference_tests::Tensor({1, 1, 5, 4}, IN_ET, std::vector<T>{0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                                                                       0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
                           "space_to_batch_4x4"),
    };
    return batchToSpaceParams;
}

std::vector<SpaceToBatchParams> generateCombinedParams() {
    const std::vector<std::vector<SpaceToBatchParams>> batchToSpaceTypeParams{
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
    std::vector<SpaceToBatchParams> combinedParams;

    for (const auto& params : batchToSpaceTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToBatch_With_Hardcoded_Refs,
                         ReferenceSpaceToBatchLayerTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceSpaceToBatchLayerTest::getTestCaseName);
}  // namespace
