// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

namespace LayerTestsDefinitions {

using ConfigMap = typename std::map<std::string, std::string>;

using NonZeroLayerTestParamsSet = typename std::tuple<
    ov::test::InputShape,                 // Input tensors shape
    InferenceEngine::Precision,           // Input precision
    std::string,                          // Device name
    ConfigMap>;                           // Additional network configuration

class NonZeroLayerTest : public testing::WithParamInterface<NonZeroLayerTestParamsSet>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NonZeroLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

protected:
    size_t startFrom = 0, range = 10;
};

}  // namespace LayerTestsDefinitions
