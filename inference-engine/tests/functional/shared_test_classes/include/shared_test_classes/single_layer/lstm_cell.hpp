// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using LSTMCellParams = typename std::tuple<
        bool,                              // using decompose to sub-ops transformation
        size_t,                            // batch
        size_t,                            // hidden size
        size_t,                            // input size
        std::vector<std::string>,          // activations
        float,                             // clip
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class LSTMCellTest : public testing::WithParamInterface<LSTMCellParams >,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMCellParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
