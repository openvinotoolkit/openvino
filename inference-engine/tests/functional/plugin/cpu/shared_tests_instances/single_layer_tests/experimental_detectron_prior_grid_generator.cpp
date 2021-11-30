// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/experimental_detectron_prior_grid_generator.hpp"

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {

const std::vector<ov::test::subgraph::ExperimentalDetectronPriorGridGeneratorTestParam> params = {
    // flatten = true (output tensor is 2D)
    {
        {true, 0, 0, 4.0f, 4.0f},
        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}})
    },
    {
        {true, 3, 6, 64.0f, 64.0f},
        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}})
    },
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
    {
        {false, 5, 3, 32.0f, 32.0f},
        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}})
    },
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

const std::vector<std::pair<std::string, std::vector<ov::runtime::Tensor>>> inputTensors = {};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalROI_static, ExperimentalDetectronPriorGridGeneratorLayerTest,
     ::testing::Combine(
        ::testing::ValuesIn(params),
         ::testing::Values(ov::element::Type_t::f32),
         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
     ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName);

} // namespace
