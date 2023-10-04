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
        ngraph::Shape,              // input shape
        ngraph::op::PadType,
        std::string                 // Device name
        > ConvStridesOptParams;

class ConvStridesOpt
        : public testing::WithParamInterface<ConvStridesOptParams>,
          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvStridesOptParams> &obj);

protected:
    void SetUp() override;
};
} // namespace SubgraphTestsDefinitions
