// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/normalize_l2.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<float> eps = {1e-6f, 1e-4f, 1e-2f, 0.1};


const std::vector<ngraph::op::EpsMode> epsMode = {
        ngraph::op::EpsMode::ADD,
        ngraph::op::EpsMode::MAX,
};

/* ============= 2D ============= */
const std::vector<std::vector<int64_t>> axes_2D = {
        {},
        {1},

        // Unsupported by CPU
        {0},
        {0, 1},
};

const auto normL2params_2D = testing::Combine(
        testing::ValuesIn(axes_2D),
        testing::ValuesIn(eps),
        testing::ValuesIn(epsMode),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{5, 3}})),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2_2D,
        NormalizeL2LayerTest,
        normL2params_2D,
        NormalizeL2LayerTest::getTestCaseName
);

/* ============= 3D ============= */
const std::vector<std::vector<int64_t>> axes_3D = {
        {},
        {1},
        {1, 2},

        // Unsupported by CPU
        {0},
        {0, 1},
        {2, 1},
        {0, 1, 2}
};

const auto normL2params_3D = testing::Combine(
        testing::ValuesIn(axes_3D),
        testing::ValuesIn(eps),
        testing::ValuesIn(epsMode),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 5, 3}})),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2_3D,
        NormalizeL2LayerTest,
        normL2params_3D,
        NormalizeL2LayerTest::getTestCaseName
);

/* ============= 4D ============= */
const std::vector<std::vector<int64_t>> axes_4D = {
        {},
        {1},
        {1, 2, 3},

        // Unsupported by CPU
        {0},
        {0, 1},
        {1, 2},
        {2, 3},
        {3, 1, 2},
        {0, 1, 2, 3}
};

const auto normL2params_4D = testing::Combine(
        testing::ValuesIn(axes_4D),
        testing::ValuesIn(eps),
        testing::ValuesIn(epsMode),
        testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 3, 10, 5}})),
        testing::ValuesIn(netPrecisions),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_NormalizeL2_4D,
        NormalizeL2LayerTest,
        normL2params_4D,
        NormalizeL2LayerTest::getTestCaseName
);
}  // namespace
