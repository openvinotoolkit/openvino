// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>

namespace SubgraphTestsDefinitions {

typedef std::tuple<
    ngraph::element::Type, // Input type
    std::string            // Device name
> parameterShapeOfResultParams;

class ParameterShapeOfResultSubgraphTest : public testing::WithParamInterface<parameterShapeOfResultParams>,
                                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<parameterShapeOfResultParams> obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
