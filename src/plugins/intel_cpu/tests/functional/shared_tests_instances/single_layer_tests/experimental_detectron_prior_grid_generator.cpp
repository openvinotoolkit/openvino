// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/experimental_detectron_prior_grid_generator.hpp"
#include "common_test_utils/data_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {

const std::vector<ov::test::subgraph::ExperimentalDetectronPriorGridGeneratorTestParam> params = {
    // flatten = true (output tensor is 2D)
    {
        {true, 0, 0, 4.0f, 4.0f},
        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}})
    },
    // task #72587
    //{
    //    {true, 3, 6, 64.0f, 64.0f},
    //    ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}})
    //},
    {
        {true, 0, 0, 4.0f, 4.0f},
        {
            // priors
            {{-1, -1}, {{3, 4}, {3, 4}}},
            // feature_map
            {{-1, -1, -1, -1}, {{1, 16, 4, 5}, {1, 16, 100, 100}}},
            // im_data
            {{-1, -1, -1, -1}, {{1, 3, 100, 200}, {1, 3, 100, 200}}}
        }
    },
    // flatten = false (output tensor is 4D)
    {
        {false, 0, 0, 8.0f, 8.0f},
        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 3, 7}, {1, 3, 100, 200}})
    },
    // task #72587
    //{
    //    {false, 5, 3, 32.0f, 32.0f},
    //    ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}})
    //},
    {
        {false, 0, 0, 8.0f, 8.0f},
        {
            // priors
            {{-1, -1}, {{3, 4}, {3, 4}}},
            // feature_map
            {{-1, -1, -1, -1}, {{1, 16, 3, 7}, {1, 16, 100, 100}}},
            // im_data
            {{-1, -1, -1, -1}, {{1, 3, 100, 200}, {1, 3, 100, 200}}}
        }
    }
};

std::vector<std::pair<std::string, std::vector<ov::Tensor>>> inputTensors = {
    {
        "test#1",
        {
            ov::test::utils::create_tensor<float>(
                    ov::element::f32,
                    ov::Shape{3, 4},
                    {-24.5, -12.5, 24.5, 12.5, -16.5, -16.5, 16.5, 16.5, -12.5, -24.5, 12.5, 24.5})
        }
    },
    {
        "test#2",
        {
            ov::test::utils::create_tensor<float>(
                    ov::element::f32,
                    ov::Shape{3, 4},
                    {-44.5, -24.5, 44.5, 24.5, -32.5, -32.5, 32.5, 32.5, -24.5, -44.5, 24.5, 44.5})
        }
    },
    {
        "test#3",
        {
            ov::test::utils::create_tensor<float>(
                    ov::element::f32,
                    ov::Shape{3, 4},
                    {-364.5, -184.5, 364.5, 184.5, -256.5, -256.5, 256.5, 256.5, -180.5, -360.5, 180.5, 360.5})
        }
    },
    {
        "test#4",
        {
            ov::test::utils::create_tensor<float>(
                    ov::element::f32,
                    ov::Shape{3, 4},
                    {-180.5, -88.5, 180.5, 88.5, -128.5, -128.5, 128.5, 128.5, -92.5, -184.5, 92.5, 184.5})
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronPriorGridGenerator, ExperimentalDetectronPriorGridGeneratorLayerTest,
     ::testing::Combine(
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(inputTensors),
        ::testing::Values(ov::element::Type_t::f32),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
     ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName);

} // namespace
