// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <array>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {
typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::array<size_t, 4>,              // Input shape
    std::array<size_t, 2>,              // Kernel shape
    size_t,                             // Output channels
    std::map<std::string, std::string>  // Configuration
> PermConvPermConcatParams;

class PermConvPermConcat : public testing::WithParamInterface<PermConvPermConcatParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PermConvPermConcatParams>& obj);

protected:
    void SetUp() override;
    void Run() override;
};
}  // namespace SubgraphTestsDefinitions
