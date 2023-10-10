// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

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
    static std::string getTestCaseName(const testing::TestParamInfo<parameterShapeOfResultParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
