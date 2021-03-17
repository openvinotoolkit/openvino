// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        InferenceEngine::SizeVector, // Input shapes
        InferenceEngine::Precision,  // Input precision
        bool,                        // Across channels
        bool,                        // Normalize variance
        double,                      // Epsilon
        std::string> mvnParams;      // Device name

class MvnLayerTest : public testing::WithParamInterface<mvnParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mvnParams>& obj);

protected:
    void SetUp() override;
};

typedef std::tuple<
        InferenceEngine::SizeVector, // Input shapes
        InferenceEngine::Precision,  // Data precision
        InferenceEngine::Precision,  // Axes precision
        std::vector<int>,            // Axes
        bool,                        // Normalize variance
        float,                       // Epsilon
        std::string,                 // Epsilon mode
        std::string                  // Device name
    > mvn6Params;

class Mvn6LayerTest : public testing::WithParamInterface<mvn6Params>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mvn6Params>& obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
