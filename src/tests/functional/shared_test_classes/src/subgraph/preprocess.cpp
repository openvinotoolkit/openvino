// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include "ov_models/preprocess/preprocess_builders.hpp"
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
    result << "Func=" << func.m_name << "_";
    result << "Device=" << targetName << "";
    return result.str();
}

void PrePostProcessTest::SetUp() {
    preprocess_func func;
    std::tie(func, targetDevice) = GetParam();
    function = func.m_function();
    rel_threshold = func.m_accuracy;
    functionRefs = ngraph::clone_function(*function);
    abs_threshold = func.m_accuracy;
    if (func.m_shapes.empty()) {
        for (const auto& input : function->inputs()) {
            func.m_shapes.push_back(input.get_shape());
        }
    }
    init_input_shapes(ov::test::static_shapes_to_test_representation(func.m_shapes));
}

TEST_P(PrePostProcessTest, CompareWithRefs) {
    run();
}

}  // namespace SubgraphTestsDefinitions
