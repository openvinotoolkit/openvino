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
        ngraph::Shape,              // Input shape
        ngraph::Shape,              // Weights shape
        ngraph::Shape,              // Const shape
        int64_t,                    // Number of ops in final function
        ngraph::element::Type,      // Network precision
        std::string                 // Device name
        > ConvMultiplyParams;

class ConvMultiply
        : public testing::WithParamInterface<ConvMultiplyParams>,
          public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvMultiplyParams> &obj);

protected:
    void SetUp() override;
};
} // namespace LayerTestsDefinitions
