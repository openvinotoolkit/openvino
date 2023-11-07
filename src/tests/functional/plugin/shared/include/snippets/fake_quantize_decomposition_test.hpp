// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace LayerTestsDefinitions {

class ActualValues {
public:
    ov::element::Type modelType;
    ngraph::Shape inputShape;
    ov::element::Type inputType;
    float zeroPoint;
    std::vector<ngraph::Shape> fakeQuantizeShapes;
};

class TestValues {
public:
    ov::element::Type modelType;
    ngraph::Shape inputShape;
    ov::element::Type inputType;
    float zeroPoint;
    std::vector<ngraph::Shape> fakeQuantizeShapes;
};

typedef std::tuple<
    TestValues,                 // test values
    std::pair<std::shared_ptr<ngraph::Node>, std::pair<std::string, std::string>>,   // operation
    std::pair<size_t, size_t>,  // number of nodes
    std::string                 // target device
> testsParams;

class FakeQuantizeDecompositionTest : public testing::WithParamInterface<testsParams>, virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<testsParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
