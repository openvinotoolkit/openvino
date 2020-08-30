// Copyright (C) 2019 Intel Corporation
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

using RNNCellParams = typename std::tuple<
        size_t,                            // batch
        size_t,                            // hidden size
        size_t,                            // input size
        std::vector<std::string>,          // activations
        std::vector<float>,                // activations_alpha
        std::vector<float>,                // activations_beta
        float,                             // clip
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name

class RNNCellTest : public testing::WithParamInterface<RNNCellParams >,
                        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RNNCellParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
