// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        bool,                               // SpecialZero
        InferenceEngine::Precision,         // Network precision
        std::vector<size_t>,                // Input shapes
        std::vector<size_t>,                // OutForm Shapes
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> reshapeParams;

class ReshapeLayerTest : public testing::WithParamInterface<reshapeParams>,
                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<reshapeParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions