// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/single_layer/shape_of.hpp"

#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

class ReluShapeOfSubgraphTest : public testing::WithParamInterface<LayerTestsDefinitions::shapeOfParamsCommon>,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LayerTestsDefinitions::shapeOfParamsCommon>& obj);
protected:
    void SetUp() override;
};
}  // namespace SubgraphTestsDefinitions
