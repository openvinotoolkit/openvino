// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include <ngraph/shape.hpp>
#include <ngraph/node.hpp>

namespace LayerTestsDefinitions {

typedef std::tuple<
        ngraph::NodeTypeInfo,       // Convolution type
        std::tuple<
            ngraph::NodeTypeInfo,   // Eltwise type
            int64_t                 // Expected number of ops
        >,
        ngraph::Shape,              // Input shape
        ngraph::Shape,              // Weights shape
        ngraph::Shape,              // Const shape
        ngraph::element::Type,      // Network precision
        std::string                 // Device name
        > ConvEltwiseFusionParams;

class ConvEltwiseFusion
        : public testing::WithParamInterface<ConvEltwiseFusionParams>,
          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvEltwiseFusionParams> &obj);

protected:
    void SetUp() override;
};
} // namespace LayerTestsDefinitions
