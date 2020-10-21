// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ngraph/op/util/attr_types.hpp>
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {
enum LOOP_IN_TYPE {
    INVARIANT,
    MERGED
};

using LoopParams = typename std::tuple<
        bool,                                                              // ExecuteFirstIteration
        bool,                                                              // BodyCondition is a constant?
        bool,                                                              // BodyCondition value, if it is a Const
        int64_t,                                                           // TripCount, -1 means infinity
        std::vector<std::pair<std::vector<size_t>, LOOP_IN_TYPE>>,         // inputs
        InferenceEngine::Precision,                                        // Network precision
        std::string>;                                                      // Device name

class LoopTest : public testing::WithParamInterface<LoopParams>,
                     virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LoopParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
