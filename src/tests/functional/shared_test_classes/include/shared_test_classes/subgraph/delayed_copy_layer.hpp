// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,         // Network precision
        std::string,                        // Device name
        std::map<std::string, std::string>, // Configuration
        size_t                              // Memory layer size
> DelayedCopyTuple;

class DelayedCopyTestBase
       : public testing::WithParamInterface<DelayedCopyTuple>,
         virtual public LayerTestsUtils::LayerTestsCommon {
private:
    void InitMemory();
    virtual void switchToNgraphFriendlyModel() = 0;
protected:
    void Run() override;
    void LoadNetwork() override;
    void Infer() override;
    std::vector<float> memory_init;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DelayedCopyTuple> &obj);
};

class DelayedCopyTest : public DelayedCopyTestBase {
private:
    void switchToNgraphFriendlyModel() override;
protected:
    void SetUp() override;
};

class DelayedCopyAfterReshapeWithMultipleConnTest : public DelayedCopyTestBase {
private:
    void switchToNgraphFriendlyModel() override;
protected:
    void SetUp() override;
};

} // namespace SubgraphTestsDefinitions
