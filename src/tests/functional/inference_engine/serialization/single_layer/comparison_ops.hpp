// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/comparison.hpp"

struct ComparisionOpsData {
    const std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes;
    const std::vector<InferenceEngine::Precision> inputsPrecisions;
    const std::vector<ngraph::helpers::InputLayerType> secondInputTypes;
    const std::map<std::string, std::string> additional_config;
    const ngraph::helpers::ComparisonTypes opType;
    const InferenceEngine::Precision ieInputPrecision;
    const InferenceEngine::Precision ieOutputPrecision;
    const std::string deviceName;
};
