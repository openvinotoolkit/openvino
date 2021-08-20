// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I32,
};

const std::vector<ShapeRelatedParams> shapeRelatedParams = {
        { { {1, 4, 5, 6}, false }, { {1, 4, 6, 4}, false } },
        { { {4, 5, 6}, false }, { {6, 3}, false } },
        { { {9, 9, 9}, false }, { {9, 9}, false } },
        { { {1, 2, 3}, false }, { {1, 10, 3}, true } },
        { { {1, 2, 3}, false }, { {1, 3, 10}, false } },
        { { {1, 2, 3}, false }, { {1, 1, 3, 2}, false } },
        { { {1, 3, 2, 4}, false }, { {2, 1, 4, 2}, false } },
        { { {2, 1, 2, 4}, false }, { {1, 3, 4, 2}, false } },
        { { {3, 2, 4}, false }, { {3, 1, 4, 2}, false } },
        { { {3, 1, 4, 2}, false }, { {1, 2, 4}, false } },
        { { {3, 2, 4, 2}, false }, { {1, 2, 4}, false } },
        { { {2, 1, 2, 3}, true }, { {3, 2, 4}, false } },
        { { {2, 1, 3, 2}, false }, { {3, 4, 2}, true } },
        { { {2, 1, 2, 3}, true }, { {3, 4, 2}, true } },
        { { {3}, false }, { {2, 2, 3, 1}, false } },
        { { {3}, false }, { {2, 2, 1, 3}, true } },
        { { {2, 2, 1, 3}, false }, { {3}, false } },
        { { {1, 1}, false }, { {1, 4}, false } },
        { { {1, 4}, false }, { {4, 5}, false } },
        { { {2, 1}, false }, { {1, 16}, false } },
        { { {1, 3}, false }, { {3, 8}, false } },
        { { {1, 8}, false }, { {1, 8}, true } },
        { { {1, 8}, false }, { {4, 8}, true } },
        { { {2, 16}, false }, { {1, 16}, true } },
        { { {1, 16}, false }, { {3, 16}, true } },
        { { {1, 18}, false }, { {1, 18}, true } },
        { { {1, 22}, false }, { {3, 22}, true } },
        { { {3, 22}, false }, { {3, 22}, true } },
        { { {3, 50}, false }, { {3, 50}, true } },
        { { {1, 1}, false }, { {1, 16}, false } },
        { { {1, 1}, false }, { {1, 20}, false } },
        { { {1, 10}, false }, { {10, 20}, false } },
        { { {3, 1}, false }, { {1, 18}, false } },
        { { {8, 2}, false }, { {2, 100}, false } },
        { { {8, 2}, false }, { {2, 10}, false } },
        { { {8, 10}, false }, { {10, 10}, false } },
        { { {8, 8}, false }, { {8, 10}, false } },
        { { {1, 8}, false }, { {8, 1}, false } },
        { { {8, 8}, false }, { {8, 1}, false } },
        { { {8, 13}, false }, { {13, 1}, false } },
        { { {1, 5}, false }, { {10, 5}, true } },
        { { {1, 5}, false }, { {5}, false } },
        { { {5}, false }, { {5, 1}, false } },
        { { {5}, false }, { {5}, false } },
        { { {5}, true }, { {5}, true } }
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_MatMul, MatMulTest,
        ::testing::Combine(
                ::testing::ValuesIn(shapeRelatedParams),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        MatMulTest::getTestCaseName);

} // namespace

