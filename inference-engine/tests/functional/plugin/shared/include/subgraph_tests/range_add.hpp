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

#include "single_layer_tests/range.hpp"

namespace LayerTestsDefinitions {

// ------------------------------ V0 ------------------------------

class RangeAddSubgraphTest : public testing::WithParamInterface<RangeParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RangeParams> obj);
protected:
    void SetUp() override;
};

// ------------------------------ V4 ------------------------------

class RangeNumpyAddSubgraphTest : public testing::WithParamInterface<RangeParams>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RangeParams> obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
