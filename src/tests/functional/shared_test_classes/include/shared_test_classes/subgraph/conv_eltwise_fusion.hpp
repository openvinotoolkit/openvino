// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
#include <ngraph/shape.hpp>
#include <ngraph/node.hpp>

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::tuple<
            ngraph::NodeTypeInfo,   // Convolution type
            size_t                  // Number of inputs
        >,
        ngraph::NodeTypeInfo,       // Eltwise type
        bool,                       // Is the test negative or not
        ngraph::Shape,              // Input shape
        ngraph::Shape,              // Weights shape
        ngraph::Shape,              // Const shape
        ngraph::element::Type,      // Network precision
        std::string                 // Device name
        > ConvEltwiseFusionParams;

class ConvEltwiseFusion
        : public testing::WithParamInterface<ConvEltwiseFusionParams>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvEltwiseFusionParams> &obj);

protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions
