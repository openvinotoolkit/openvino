// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"


using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

using ConditionParams = typename std::tuple<>;


class ConditionLayerGPUTest : public testing::WithParamInterface<ConditionParams>,
                        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConditionParams>& obj) {
        return "";
    }

protected:
    void SetUp() override {
    }
};

TEST_P(ConditionLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

}   // namespace GPULayerTestsDefinitions