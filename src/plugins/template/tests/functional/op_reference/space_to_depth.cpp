// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_depth.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct SpaceToDepthParams {
    SpaceToDepthParams(const reference_tests::Tensor& dataTensor,
                       const std::string mode,
                       const int32_t blockSize,
                       const reference_tests::Tensor& expectedTensor,
                       const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          mode(mode),
          blockSize(blockSize),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    std::string mode;
    int32_t blockSize;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceSpaceToDepthLayerTest : public testing::TestWithParam<SpaceToDepthParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SpaceToDepthParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
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
    static std::shared_ptr<Model> CreateFunction(const SpaceToDepthParams& params) {
        const auto mode = params.mode == "DEPTH_FIRST" ? op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST
                                                       : op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto SpaceToDepth = std::make_shared<op::v0::SpaceToDepth>(data, mode, params.blockSize);
        return std::make_shared<Model>(NodeVector{SpaceToDepth}, ParameterVector{data});
    }
};

TEST_P(ReferenceSpaceToDepthLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SpaceToDepthParams> generateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<SpaceToDepthParams> params{
        // space_to_depth_block_first_K2_BS2
        SpaceToDepthParams(
            reference_tests::Tensor({1, 2, 4, 4}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31}),
            "BLOCKS_FIRST",
            2,
            reference_tests::Tensor({1, 8, 2, 2}, IN_ET, std::vector<T>{0,  2,  8,  10, 16, 18, 24, 26, 1,  3,  9,
                                                                        11, 17, 19, 25, 27, 4,  6,  12, 14, 20, 22,
                                                                        28, 30, 5,  7,  13, 15, 21, 23, 29, 31}),
            "space_to_depth_block_first_K2_BS2"),

        // space_to_depth_block_first_K2_BS3
        SpaceToDepthParams(
            reference_tests::Tensor({1, 2, 6, 3},
                                    IN_ET,
                                    std::vector<T>{0, 4, 8,  12, 16, 20, 24, 28, 32, 1, 5, 9,  13, 17, 21, 25, 29, 33,
                                                   2, 6, 10, 14, 18, 22, 26, 30, 34, 3, 7, 11, 15, 19, 23, 27, 31, 35}),
            "BLOCKS_FIRST",
            3,
            reference_tests::Tensor({1, 18, 2, 1}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,
                                                                         9,  10, 11, 12, 13, 14, 15, 16, 17,
                                                                         18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                         27, 28, 29, 30, 31, 32, 33, 34, 35}),
            "space_to_depth_block_first_K2_BS3"),

        // space_to_depth_block_first_K1_BS3
        SpaceToDepthParams(
            reference_tests::Tensor({1, 2, 6}, IN_ET, std::vector<T>{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11}),
            "BLOCKS_FIRST",
            3,
            reference_tests::Tensor({1, 6, 2}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
            "space_to_depth_block_first_K1_BS3"),

        // space_to_depth_depth_first_K2_BS2
        SpaceToDepthParams(
            reference_tests::Tensor({1, 2, 4, 4}, IN_ET, std::vector<T>{0,  16, 2,  18, 1,  17, 3,  19, 8,  24, 10,
                                                                        26, 9,  25, 11, 27, 4,  20, 6,  22, 5,  21,
                                                                        7,  23, 12, 28, 14, 30, 13, 29, 15, 31}),
            "DEPTH_FIRST",
            2,
            reference_tests::Tensor({1, 8, 2, 2}, IN_ET, std::vector<T>{0,  2,  8,  10, 16, 18, 24, 26, 1,  3,  9,
                                                                        11, 17, 19, 25, 27, 4,  6,  12, 14, 20, 22,
                                                                        28, 30, 5,  7,  13, 15, 21, 23, 29, 31}),
            "space_to_depth_depth_first_K2_BS2"),

        // space_to_depth_depth_first_K2_BS3
        SpaceToDepthParams(
            reference_tests::Tensor({1, 2, 6, 3}, IN_ET, std::vector<T>{0,  2,  4,  6,  8,  10, 12, 14, 16,
                                                                        1,  3,  5,  7,  9,  11, 13, 15, 17,
                                                                        18, 20, 22, 24, 26, 28, 30, 32, 34,
                                                                        19, 21, 23, 25, 27, 29, 31, 33, 35}),
            "DEPTH_FIRST",
            3,
            reference_tests::Tensor({1, 18, 2, 1}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,
                                                                         9,  10, 11, 12, 13, 14, 15, 16, 17,
                                                                         18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                                         27, 28, 29, 30, 31, 32, 33, 34, 35}),
            "space_to_depth_depth_first_K2_BS3"),

        // space_to_depth_depth_first_K1_BS3
        SpaceToDepthParams(
            reference_tests::Tensor({1, 2, 6}, IN_ET, std::vector<T>{0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11}),
            "DEPTH_FIRST",
            3,
            reference_tests::Tensor({1, 6, 2}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
            "space_to_depth_depth_first_K1_BS3"),
    };
    return params;
}

std::vector<SpaceToDepthParams> generateCombinedParams() {
    const std::vector<std::vector<SpaceToDepthParams>> generatedParams{
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
    std::vector<SpaceToDepthParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepth_With_Hardcoded_Refs,
                         ReferenceSpaceToDepthLayerTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceSpaceToDepthLayerTest::getTestCaseName);
}  // namespace
