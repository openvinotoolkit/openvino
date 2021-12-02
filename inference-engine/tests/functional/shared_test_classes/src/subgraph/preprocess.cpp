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
    rel_threshold = std::get<2>(func);
    functionRefs = ngraph::clone_function(*function);
    abs_threshold = std::get<2>(func);
    auto static_shapes = std::get<3>(func);
    if (static_shapes.empty()) {
        for (const auto& input : function->inputs()) {
            static_shapes.push_back(input.get_shape());
        }
    }
    init_input_shapes(ov::test::static_shapes_to_test_representation({static_shapes}));
}

TEST_P(PrePostProcessTest, CompareWithRefs) {
    run();
}

}  // namespace SubgraphTestsDefinitions
