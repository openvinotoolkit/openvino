// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using bucketizeParamsTuple = std::tuple<
    InferenceEngine::SizeVector,    // Data shape
    InferenceEngine::SizeVector,    // Buckets shape
    bool,                           // Right edge of interval
    InferenceEngine::Precision,     // Data input precision
    InferenceEngine::Precision,     // Buckets input precision
    InferenceEngine::Precision,     // Output precision
    std::string>;                   // Device name

class BucketizeLayerTest : public testing::WithParamInterface<bucketizeParamsTuple>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<bucketizeParamsTuple>& obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions
