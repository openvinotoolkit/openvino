// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

#include "single_layer_tests/shape_of.hpp"

namespace LayerTestsDefinitions {

class ReluShapeOfSubgraphTest : public testing::WithParamInterface<shapeOfParams>,
        public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<shapeOfParams> obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions