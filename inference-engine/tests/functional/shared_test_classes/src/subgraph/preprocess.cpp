// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include "ngraph_functions/preprocess/preprocess_builders.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"

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
    functionRefs = ngraph::clone_function(*function);
    abs_threshold = std::get<2>(func);
}

TEST_P(PrePostProcessTest, CompareWithRefs) {
    Run();
}


std::string PrePostProcessTestDynamic::getTestCaseName(
        const testing::TestParamInfo<preprocessParamsTupleDynamic> &obj) {
    std::string targetName;
    preprocess_func_dynamic func;

    std::tie(func, targetName) = obj.param;

    std::ostringstream result;
    result << "Func=" << std::get<1>(func) << "_";
    result << "Device=" << targetName << "";
    return result.str();
}

void PrePostProcessTestDynamic::SetUp() {
    preprocess_func_dynamic func;
    std::tie(func, targetDevice) = GetParam();
    function = (std::get<0>(func))();
    rel_threshold = std::get<2>(func);
    targetStaticShapes = std::vector<std::vector<ngraph::Shape>>{{std::get<3>(func)}};
    functionRefs = ngraph::clone_function(*function);
    abs_threshold = std::get<2>(func);
}

TEST_P(PrePostProcessTestDynamic, CompareWithRefs) {
    run();
}

}  // namespace SubgraphTestsDefinitions
