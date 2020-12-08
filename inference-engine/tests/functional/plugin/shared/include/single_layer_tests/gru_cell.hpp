// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using GRUCellParams = typename std::tuple<
        bool,                              // using decompose to sub-ops transformation
        size_t,                            // batch
        size_t,                            // hidden size
        size_t,                            // input size
        std::vector<std::string>,          // activations
        float,                             // clip
        bool,                              // linear_before_reset
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class GRUCellTest : public testing::WithParamInterface<GRUCellParams >,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GRUCellParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
