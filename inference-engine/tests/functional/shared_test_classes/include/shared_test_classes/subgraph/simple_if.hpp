// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

using SimpleIfParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>,    // Input shapes
        InferenceEngine::Precision,          // Network precision
        bool,                                // If condition
        std::string                          // Device name
>;

class SimpleIfTest:
        public testing::WithParamInterface<SimpleIfParamsTuple>,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SimpleIfParamsTuple> &obj);
protected:
    void SetUp() override;
};

class SimpleIf2OutTest : public SimpleIfTest {
protected:
    void SetUp() override;
};

class SimpleIfNotConstConditionTest : public SimpleIfTest {
public:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

protected:
    void SetUp() override;

    bool condition;
};

}  // namespace SubgraphTestsDefinitions
