// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/depth_to_space.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct DepthToSpaceParams {
    DepthToSpaceParams(const reference_tests::Tensor& dataTensor,
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

class ReferenceDepthToSpaceLayerTest : public testing::TestWithParam<DepthToSpaceParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<DepthToSpaceParams>& obj) {
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
    static std::shared_ptr<Model> CreateFunction(const DepthToSpaceParams& params) {
        const auto mode = params.mode == "DEPTH_FIRST" ? op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST
                                                       : op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto depthToSpace = std::make_shared<op::v0::DepthToSpace>(data, mode, params.blockSize);
        return std::make_shared<Model>(NodeVector{depthToSpace}, ParameterVector{data});
    }
};

TEST_P(ReferenceDepthToSpaceLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<DepthToSpaceParams> generateDepthToSpaceParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<DepthToSpaceParams> depthToSpaceParams{
        // depth_to_space_block_first_K1_BS2
        DepthToSpaceParams(
            reference_tests::Tensor({1, 8, 2},
                                    IN_ET,
                                    std::vector<T>{0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27}),
            "BLOCKS_FIRST",
            2,
            reference_tests::Tensor({1, 4, 4},
                                    IN_ET,
                                    std::vector<T>{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27}),
            "depth_to_space_block_first_K1_BS2"),

        // depth_to_space_block_first_K2_BS2
        DepthToSpaceParams(
            reference_tests::Tensor({1, 8, 2, 2}, IN_ET, std::vector<T>{0,  2,  8,  10, 16, 18, 24, 26, 1,  3,  9,
                                                                        11, 17, 19, 25, 27, 4,  6,  12, 14, 20, 22,
                                                                        28, 30, 5,  7,  13, 15, 21, 23, 29, 31}),
            "BLOCKS_FIRST",
            2,
            reference_tests::Tensor({1, 2, 4, 4}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31}),
            "depth_to_space_block_first_K2_BS2"),

        // depth_to_space_block_first_K2_BS4
        DepthToSpaceParams(
            reference_tests::Tensor({1, 16, 2, 1}, IN_ET, std::vector<T>{0,  16, 1,  17, 2,  18, 3,  19, 4,  20, 5,
                                                                         21, 6,  22, 7,  23, 8,  24, 9,  25, 10, 26,
                                                                         11, 27, 12, 28, 13, 29, 14, 30, 15, 31}),
            "BLOCKS_FIRST",
            4,
            reference_tests::Tensor({1, 1, 8, 4}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31}),
            "depth_to_space_block_first_K2_BS4"),

        // depth_to_space_depth_first_1K_BS2
        DepthToSpaceParams(
            reference_tests::Tensor({1, 8, 2},
                                    IN_ET,
                                    std::vector<T>{0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15}),
            "DEPTH_FIRST",
            2,
            reference_tests::Tensor({1, 4, 4},
                                    IN_ET,
                                    std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
            "depth_to_space_depth_first_1K_BS2"),

        // depth_to_space_depth_first_2K_BS2
        DepthToSpaceParams(
            reference_tests::Tensor({1, 8, 2, 2}, IN_ET, std::vector<T>{0,  2,  8,  10, 16, 18, 24, 26, 1,  3,  9,
                                                                        11, 17, 19, 25, 27, 4,  6,  12, 14, 20, 22,
                                                                        28, 30, 5,  7,  13, 15, 21, 23, 29, 31}),
            "DEPTH_FIRST",
            2,
            reference_tests::Tensor({1, 2, 4, 4}, IN_ET, std::vector<T>{0,  16, 2,  18, 1,  17, 3,  19, 8,  24, 10,
                                                                        26, 9,  25, 11, 27, 4,  20, 6,  22, 5,  21,
                                                                        7,  23, 12, 28, 14, 30, 13, 29, 15, 31}),
            "depth_to_space_depth_first_2K_BS2"),

        // depth_to_space_depth_first_2K_BS4
        DepthToSpaceParams(
            reference_tests::Tensor({1, 16, 2, 1}, IN_ET, std::vector<T>{0,  16, 1,  17, 2,  18, 3,  19, 4,  20, 5,
                                                                         21, 6,  22, 7,  23, 8,  24, 9,  25, 10, 26,
                                                                         11, 27, 12, 28, 13, 29, 14, 30, 15, 31}),
            "DEPTH_FIRST",
            4,
            reference_tests::Tensor({1, 1, 8, 4}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                                        22, 23, 24, 25, 26, 27, 28, 29, 30, 31}),
            "depth_to_space_depth_first_2K_BS4"),
    };
    return depthToSpaceParams;
}

std::vector<DepthToSpaceParams> generateDepthToSpaceCombinedParams() {
    const std::vector<std::vector<DepthToSpaceParams>> depthToSpaceTypeParams{
        generateDepthToSpaceParams<element::Type_t::i8>(),
        generateDepthToSpaceParams<element::Type_t::i16>(),
        generateDepthToSpaceParams<element::Type_t::i32>(),
        generateDepthToSpaceParams<element::Type_t::i64>(),
        generateDepthToSpaceParams<element::Type_t::u8>(),
        generateDepthToSpaceParams<element::Type_t::u16>(),
        generateDepthToSpaceParams<element::Type_t::u32>(),
        generateDepthToSpaceParams<element::Type_t::u64>(),
        generateDepthToSpaceParams<element::Type_t::bf16>(),
        generateDepthToSpaceParams<element::Type_t::f16>(),
        generateDepthToSpaceParams<element::Type_t::f32>(),
        generateDepthToSpaceParams<element::Type_t::f64>(),
    };
    std::vector<DepthToSpaceParams> combinedParams;

    for (const auto& params : depthToSpaceTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace_With_Hardcoded_Refs,
                         ReferenceDepthToSpaceLayerTest,
                         testing::ValuesIn(generateDepthToSpaceCombinedParams()),
                         ReferenceDepthToSpaceLayerTest::getTestCaseName);
}  // namespace
