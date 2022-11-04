// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/generate_proposals.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "benchmark.hpp"

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {

const std::vector<float> min_size = { 1 };
const std::vector<float> nms_threshold = { 0.699999988079071 };
const std::vector<int64_t> post_nms_count = { 6 };
const std::vector<int64_t> pre_nms_count = { 1000 };

const std::vector<std::pair<std::string, std::vector<ov::Tensor>>> inputTensors0 = {
    {
        "anchor_9",
        {
            // 3
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 3}, 255, 0, 1, 17),
            // 2 x 6 x 3 x 4 = 144
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 6, 9, 4}, 255, 0, 1, 21),
            // 2 x 36 x 2 x 6 = 144 * 2
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 36, 2, 6}, 255, 0, 1, 11),
            // {2 x 9 x 2 x 6} = 36 * 2
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 9, 2, 6}, 255, 0, 1, 8)


        }
    },
};

const std::vector<std::pair<std::string, std::vector<ov::Tensor>>> inputTensors1 = {
    {
        "anchor_16",
        {
            // 3
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 3}, 255, 0, 1, 17),
            // 2 x 6 x 3 x 4 = 144
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 6, 16, 4}, 255, 0, 1, 21),
            // 2 x 64 x 2 x 6 = 144 * 2
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 64, 2, 6}, 255, 0, 1, 11),
            // {2 x 16 x 2 x 6} = 36 * 2
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 16, 2, 6}, 255, 0, 1, 8)
        }
    },
};

const std::vector<std::pair<std::string, std::vector<ov::Tensor>>> inputTensors2 = {
    {
        "anchor_256",
        {
            // 3
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 3}, 255, 0, 1, 17),
            // 2 x 6 x 3 x 4 = 144
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 6, 256, 4}, 255, 0, 1, 21),
            // 2 x 1024 x 2 x 6 = 144 * 2
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 1024, 2, 6}, 255, 0, 1, 11),
            // {2 x 256 x 2 x 6} = 36 * 2
            ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{2, 256, 2, 6}, 255, 0, 1, 8)
        }
    },
};

const std::vector<std::vector<InputShape>> staticInputShape0 = {
    // im_info / anchors / deltas / scores
    static_shapes_to_test_representation(
        {
            {2, 3},
            {2, 6, 9, 4},
            {2, 36, 2, 6},
            {2, 9, 2, 6}
        }
    ),
};

const std::vector<std::vector<InputShape>> staticInputShape1 = {
    // im_info / anchors / deltas / scores
    static_shapes_to_test_representation(
        {
            {2, 3},
            {2, 6, 16, 4},
            {2, 64, 2, 6},
            {2, 16, 2, 6}
        }
    ),
};

const std::vector<std::vector<InputShape>> staticInputShape2 = {
    // im_info / anchors / deltas / scores
    static_shapes_to_test_representation(
        {
            {2, 3},
            {2, 6, 256, 4},
            {2, 1024, 2, 6},
            {2, 256, 2, 6}
        }
    ),
};

INSTANTIATE_TEST_SUITE_P(
    smoke_GenerateProposalsLayerTest_staticInputShape0,
    GenerateProposalsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(staticInputShape0),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(nms_threshold),
        ::testing::ValuesIn(post_nms_count),
        ::testing::ValuesIn(pre_nms_count),
        ::testing::ValuesIn(inputTensors0),
        ::testing::Values(ov::element::Type_t::f32),
        ::testing::Values(ov::element::Type_t::i32),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GenerateProposalsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_GenerateProposalsLayerTest_staticInputShape1,
        GenerateProposalsLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(staticInputShape1),
                ::testing::ValuesIn(min_size),
                ::testing::ValuesIn(nms_threshold),
                ::testing::ValuesIn(post_nms_count),
                ::testing::ValuesIn(pre_nms_count),
                ::testing::ValuesIn({true, false}),
                ::testing::ValuesIn(inputTensors1),
                ::testing::Values(ov::element::Type_t::f32),
                ::testing::Values(ov::element::Type_t::i32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        GenerateProposalsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GenerateProposalsLayerTest_staticInputShape2,
    GenerateProposalsLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(staticInputShape2),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(nms_threshold),
        ::testing::ValuesIn(post_nms_count),
        ::testing::ValuesIn(pre_nms_count),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn(inputTensors2),
        ::testing::Values(ov::element::Type_t::f32),
        ::testing::Values(ov::element::Type_t::i32),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GenerateProposalsLayerTest::getTestCaseName);

struct GenerateProposalsBenchmarkTest : ov::test::BenchmarkLayerTest<GenerateProposalsLayerTest> {};

TEST_P(GenerateProposalsBenchmarkTest, GenerateProposals_Benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run("GenerateProposals", std::chrono::milliseconds(2000), 1000);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_GenerateProposalsBenchmarkTest_staticInputShape0,
    GenerateProposalsBenchmarkTest,
    ::testing::Combine(
        ::testing::ValuesIn(staticInputShape0),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(nms_threshold),
        ::testing::ValuesIn(post_nms_count),
        ::testing::ValuesIn(pre_nms_count),
        ::testing::ValuesIn(inputTensors0),
        ::testing::Values(ov::element::Type_t::f32),
        ::testing::Values(ov::element::Type_t::i32),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GenerateProposalsBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GenerateProposalsBenchmarkTest_staticInputShape1,
    GenerateProposalsBenchmarkTest,
    ::testing::Combine(
        ::testing::ValuesIn(staticInputShape1),
        ::testing::ValuesIn(min_size),
        ::testing::ValuesIn(nms_threshold),
        ::testing::ValuesIn(post_nms_count),
        ::testing::ValuesIn(pre_nms_count),
        ::testing::ValuesIn(inputTensors1),
        ::testing::Values(ov::element::Type_t::f32),
        ::testing::Values(ov::element::Type_t::i32),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GenerateProposalsBenchmarkTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_GenerateProposalsBenchmarkTest_staticInputShape2,
        GenerateProposalsBenchmarkTest,
        ::testing::Combine(
                ::testing::ValuesIn(staticInputShape2),
                ::testing::ValuesIn(min_size),
                ::testing::ValuesIn(nms_threshold),
                ::testing::ValuesIn(post_nms_count),
                ::testing::ValuesIn(pre_nms_count),
                ::testing::ValuesIn(inputTensors2),
                ::testing::Values(ov::element::Type_t::f32),
                ::testing::Values(ov::element::Type_t::i32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        GenerateProposalsBenchmarkTest::getTestCaseName);

} // namespace
