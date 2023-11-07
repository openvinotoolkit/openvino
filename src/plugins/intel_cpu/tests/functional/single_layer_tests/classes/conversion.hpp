// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/activation.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"

using namespace InferenceEngine;
using namespace ngraph;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {
using convertLayerTestParamsSet = std::tuple<InputShape,                   // input shapes
                                        InferenceEngine::Precision,        // input precision
                                        InferenceEngine::Precision,        // output precision
                                        CPUSpecificParams>;

class ConvertCPULayerTest : public testing::WithParamInterface<convertLayerTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convertLayerTestParamsSet> obj);
    static bool isInOutPrecisionSupported(InferenceEngine::Precision inPrc, InferenceEngine::Precision outPrc);
protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

private:
    InferenceEngine::Precision inPrc, outPrc;
};

namespace Conversion {
    const std::vector<InputShape>& inShapes_4D_static();
    const std::vector<InputShape>& inShapes_4D_dynamic();
    const std::vector<InputShape>& inShapes_7D_static();
    const std::vector<InputShape>& inShapes_7D_dynamic();
    const std::vector<Precision>& precisions();
} // namespace Conversion
} // namespace CPULayerTestsDefinitions