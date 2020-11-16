// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace CPUTestUtils;

namespace LayerTestsDefinitions {

/*    graph0                      graph1
      ---------                   ---------
      |Input  |                   |Input  |
      ---------                   ---------
          |                           |
    -------------                 ---------
    | --------- |                 |Permute|
    | |Permute| |                 ---------
    | --------- |                     |
    |     |     |            -------------------
    | --------- |            |                 |
    | |Reorder| |            |           -------------
    | --------- |
    |-----------|            |           | --------- |
          |                  |           | |Permute| |
      ---------          ---------       | --------- |
      |Output |          |Reshape|       |     |     |
      ---------          ---------       | --------- |
                             |           | |Reorder| |
                             |           | --------- |
                             |           |-----------|
                             |                 |
                             |             ---------
                             |             |Permute|
                             |             ---------
                             |                 |
                             --------   --------
                                    |   |
                                  ---------
                                  |Concat |
                                  ---------
                                      |
                                  ---------
                                  |Output |
                                  ---------
*/

enum TestGraphType {
    graph0 = 0,
    graph1 = 1
};

using FusePermuteAndReorderParams = std::tuple<
        TestGraphType,                  // Test graph type
        InferenceEngine::SizeVector,    // Input shape
        InferenceEngine::Precision      // Input precision
>;

class FusePermuteAndReorderTest : public testing::WithParamInterface<FusePermuteAndReorderParams>, public CPUTestsBase,
        virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FusePermuteAndReorderParams> obj);

protected:
    void SetUp() override;
    void createGraph0();
    void createGraph1();

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision inPrec;
    TestGraphType graphType;
};

} // namespace LayerTestsDefinitions
