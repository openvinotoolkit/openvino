// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    ov::element::Type,           // Network Precision
    ngraph::Shape,               // Input 0 Shape
    ov::element::Type,           // input precision
    bool,                        // model in low precision
    std::vector<ngraph::Shape>,  // FakeQuantize shapes
    std::string                  // Target Device
> inputParams;

class ActualValues {
public:
    ov::element::Type modelType;
    ngraph::Shape inputShape;
    ov::element::Type inputType;
    float zeroPoint;
    std::vector<ngraph::Shape> fakeQuantizeShapes;
    std::string targetDevice;
};

class Operation {
public:
    std::string name;
    std::string type;
};

class ExpectedValues {
public:
    int operationsCount;
    std::vector<Operation> expectedOperations;
    std::vector<std::string> notExpectedOperationTypes;
};

class TestValues {
public:
    ActualValues actual;
    ExpectedValues expected;
};

class FakeQuantizeDecompositionTest : public testing::WithParamInterface<TestValues>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TestValues> obj);

protected:
    void GenerateInputs() override;
    void SetUp() override;
    void Run() override;
};

}  // namespace LayerTestsDefinitions
