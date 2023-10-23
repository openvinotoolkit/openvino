// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        std::vector<size_t>,                // Input shapes
        std::string                        // Device name
> shapeOfParamsCommon;

typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        InferenceEngine::Precision,          // Output precision
        std::vector<size_t>,                // Input shapes
        std::string                        // Device name
> shapeOfParams;

class ShapeOfLayerTest : public testing::WithParamInterface<shapeOfParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ParamType> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
