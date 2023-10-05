// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,                 // Network Precision
        std::string,                                // Target Device
        std::vector<size_t>,                        // Input Shapes
        std::vector<std::pair<int64_t, int64_t>>,   // Offset pairs (begin, end)
        std::map<std::string, std::string>         // Configuration
> MultiCropsToConcatParams;


class MultiCropsToConcatTest : public testing::WithParamInterface<MultiCropsToConcatParams>,
                                   public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultiCropsToConcatParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace SubgraphTestsDefinitions
