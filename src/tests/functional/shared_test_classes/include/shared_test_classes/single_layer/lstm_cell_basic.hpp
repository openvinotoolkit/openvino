// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsDefinitions {

using LSTMCellBasicParams = typename std::tuple<
        bool,                                   // using decompose to sub-ops transformation
        size_t,                                 // batch
        size_t,                                 // hidden size
        size_t,                                 // input size
        std::vector<std::string>,               // activations
        float,                                  // clip
        InferenceEngine::Precision,             // Network precision
        std::string,                            // Device name
        std::map<std::string, std::string>>;    // Config

class LSTMCellBasicTest : public testing::WithParamInterface<LSTMCellBasicParams >,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMCellBasicParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
