// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/comparison.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

static const std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
    {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
    {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
    {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
    {{1, 3, 20}, {{20}, {2, 1, 1}}},
    {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
    {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {141, 1, 3, 4}}},
    {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

static const std::vector<InferenceEngine::Precision> inputsPrecisions = {
    InferenceEngine::Precision::FP64,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::U32,
    InferenceEngine::Precision::BOOL,
};

static const std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

static const std::map<std::string, std::string> additional_config = {};