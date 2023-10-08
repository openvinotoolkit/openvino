// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<std::tuple<ov::NodeTypeInfo,  // Convolution type
                              size_t             // Number of inputs
                              >,
                   ov::NodeTypeInfo,   // Eltwise type
                   bool,               // Is the test negative or not
                   ov::Shape,          // Input shape
                   ov::Shape,          // Weights shape
                   ov::Shape,          // Const shape
                   ov::element::Type,  // Network precision
                   std::string         // Device name
                   >
    ConvEltwiseFusionParams;

class ConvEltwiseFusion : public testing::WithParamInterface<ConvEltwiseFusionParams>,
                          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvEltwiseFusionParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
