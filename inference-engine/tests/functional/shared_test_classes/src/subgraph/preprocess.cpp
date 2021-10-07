// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include "ngraph_functions/preprocess/preprocess_builders.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include <ie_ngraph_utils.hpp>

using namespace ov;
using namespace ov::preprocess;
using namespace ov::builder::preprocess;

namespace SubgraphTestsDefinitions {
std::string PrePostProcessTest::getTestCaseName(
        const testing::TestParamInfo<preprocessParamsTuple> &obj) {
    std::string targetName;
    preprocess_func func;

    std::tie(func, targetName) = obj.param;

    std::ostringstream result;
    result << "Func=" << std::get<1>(func) << "_";
    result << "Device=" << targetName << "";
    return result.str();
}

void PrePostProcessTest::SetUp() {
    preprocess_func func;
    std::tie(func, targetDevice) = GetParam();
    function = (std::get<0>(func))();
    threshold = std::get<2>(func);
    abs_threshold = std::get<2>(func);
}

TEST_P(PrePostProcessTest, CompareWithRefs) {
    Run();
}

}  // namespace SubgraphTestsDefinitions
