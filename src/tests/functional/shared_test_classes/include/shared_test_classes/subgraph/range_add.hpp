// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"

#include "shared_test_classes/single_layer/range.hpp"

namespace SubgraphTestsDefinitions {

// ------------------------------ V0 ------------------------------

class RangeAddSubgraphTest : public testing::WithParamInterface<LayerTestsDefinitions::RangeParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LayerTestsDefinitions::RangeParams>& obj);
protected:
    void SetUp() override;
};

// ------------------------------ V4 ------------------------------

class RangeNumpyAddSubgraphTest : public testing::WithParamInterface<LayerTestsDefinitions::RangeParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LayerTestsDefinitions::RangeParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
