// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using batchToSpaceParamsTuple = typename std::tuple<
        std::vector<size_t>,               // block shape
        std::vector<size_t>,               // crops begin
        std::vector<size_t>,               // crops end
        std::vector<size_t>,               // Input shapes
        InferenceEngine::Precision,        // Network precision
        std::string>;                      // Device name>;

class BatchToSpaceLayerTest : public testing::WithParamInterface<batchToSpaceParamsTuple>,
                              public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<batchToSpaceParamsTuple> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions