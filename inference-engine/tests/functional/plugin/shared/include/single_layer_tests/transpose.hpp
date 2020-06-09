// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        std::vector<size_t>,                // Input shapes
        std::vector<size_t>,                // Input order
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> transposeParams;

class TransposeLayerTest : public testing::WithParamInterface<transposeParams>,
                           public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<transposeParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
