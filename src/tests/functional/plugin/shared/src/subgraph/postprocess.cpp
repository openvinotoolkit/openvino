// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "common_test_utils/subgraph_builders/preprocess_builders.hpp"

using namespace ov;
using namespace ov::preprocess;
using namespace ov::builder::preprocess;

namespace ov {
namespace test {

std::string PostProcessTest::getTestCaseName(const testing::TestParamInfo<postprocessParamsTuple>& obj) {
    std::string targetName;
    postprocess_func func;

    std::tie(func, targetName) = obj.param;
    std::ostringstream result;
    result << "Func=" << func.m_name << "_";
    result << "Device=" << targetName << "";
    return result.str();
}

void PostProcessTest::SetUp() {
    postprocess_func func;
    std::tie(func, targetDevice) = GetParam();
    function = func.m_function();
    rel_threshold = func.m_accuracy;
    functionRefs = function->clone();
    abs_threshold = func.m_accuracy;
    if (func.m_shapes.empty()) {
        for (const auto& input : function->inputs()) {
            func.m_shapes.push_back(input.get_shape());
        }
    }
    init_input_shapes(ov::test::static_shapes_to_test_representation(func.m_shapes));
    func_name = func.m_name;
}

TEST_P(PostProcessTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
