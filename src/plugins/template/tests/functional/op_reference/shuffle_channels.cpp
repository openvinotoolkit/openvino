// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shuffle_channels.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ShuffleChannelsParams {
    ShuffleChannelsParams(const reference_tests::Tensor& dataTensor,
                          const int32_t axis,
                          const int32_t group,
                          const reference_tests::Tensor& expectedTensor,
                          const std::string& testcaseName = "")
        : dataTensor(dataTensor),
          axis(axis),
          group(group),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName) {}

    reference_tests::Tensor dataTensor;
    int32_t axis;
    int32_t group;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
};

class ReferenceShuffleChannelsLayerTest : public testing::TestWithParam<ShuffleChannelsParams>,
                                          public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data};
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ShuffleChannelsParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_axis=" << param.axis;
        result << "_group=" << param.group;
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
    static std::shared_ptr<Model> CreateFunction(const ShuffleChannelsParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto function = std::make_shared<op::v0::ShuffleChannels>(data, params.axis, params.group);
        return std::make_shared<Model>(NodeVector{function}, ParameterVector{data});
    }
};

TEST_P(ReferenceShuffleChannelsLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ShuffleChannelsParams> generateParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<ShuffleChannelsParams> params{
        // shuffle_channels_simple
        ShuffleChannelsParams(
            reference_tests::Tensor(
                {1, 15, 2, 2},
                IN_ET,
                std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}),
            1,
            5,
            reference_tests::Tensor(
                {1, 15, 2, 2},
                IN_ET,
                std::vector<T>{0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
                               4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
                               8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59}),
            "shuffle_channels_simple"),

        // shuffle_channels_negative_axis
        ShuffleChannelsParams(
            reference_tests::Tensor(
                {15, 2, 1, 2},
                IN_ET,
                std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}),
            -4,
            5,
            reference_tests::Tensor(
                {15, 2, 1, 2},
                IN_ET,
                std::vector<T>{0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
                               4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
                               8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59}),
            "shuffle_channels_negative_axis"),

        // shuffle_channels_float
        ShuffleChannelsParams(reference_tests::Tensor({6, 1, 1, 1}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5}),
                              0,
                              2,
                              reference_tests::Tensor({6, 1, 1, 1}, IN_ET, std::vector<T>{0, 3, 1, 4, 2, 5}),
                              "shuffle_channels_float"),

        // shuffle_channels_1d
        ShuffleChannelsParams(
            reference_tests::Tensor({15}, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}),
            0,
            5,
            reference_tests::Tensor({15}, IN_ET, std::vector<T>{0, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14}),
            "shuffle_channels_1d"),

        // shuffle_channels_2d
        ShuffleChannelsParams(
            reference_tests::Tensor({15, 4}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                   24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                                   48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}),
            0,
            5,
            reference_tests::Tensor({15, 4}, IN_ET, std::vector<T>{0,  1,  2,  3,  12, 13, 14, 15, 24, 25, 26, 27,
                                                                   36, 37, 38, 39, 48, 49, 50, 51, 4,  5,  6,  7,
                                                                   16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43,
                                                                   52, 53, 54, 55, 8,  9,  10, 11, 20, 21, 22, 23,
                                                                   32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59}),
            "shuffle_channels_2d"),

        // shuffle_channels_3d
        ShuffleChannelsParams(
            reference_tests::Tensor({15, 2, 2}, IN_ET, std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                                                      12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                      36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                                                      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}),
            0,
            5,
            reference_tests::Tensor({15, 2, 2}, IN_ET, std::vector<T>{0,  1,  2,  3,  12, 13, 14, 15, 24, 25, 26, 27,
                                                                      36, 37, 38, 39, 48, 49, 50, 51, 4,  5,  6,  7,
                                                                      16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43,
                                                                      52, 53, 54, 55, 8,  9,  10, 11, 20, 21, 22, 23,
                                                                      32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59}),
            "shuffle_channels_3d"),

        // shuffle_channels_5d
        ShuffleChannelsParams(
            reference_tests::Tensor(
                {2, 2, 15, 2, 2},
                IN_ET,
                std::vector<T>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,

                               0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,

                               0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,

                               0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}),
            2,
            5,
            reference_tests::Tensor(
                {2, 2, 15, 2, 2},
                IN_ET,
                std::vector<T>{0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
                               4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
                               8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59,

                               0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
                               4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
                               8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59,

                               0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
                               4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
                               8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59,

                               0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
                               4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
                               8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59}),
            "shuffle_channels_5d"),
    };
    return params;
}

std::vector<ShuffleChannelsParams> generateShuffleChannelsCombinedParams() {
    const std::vector<std::vector<ShuffleChannelsParams>> generatedParams{
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
    std::vector<ShuffleChannelsParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels_With_Hardcoded_Refs,
                         ReferenceShuffleChannelsLayerTest,
                         testing::ValuesIn(generateShuffleChannelsCombinedParams()),
                         ReferenceShuffleChannelsLayerTest::getTestCaseName);
}  // namespace
