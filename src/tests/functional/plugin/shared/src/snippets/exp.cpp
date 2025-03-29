// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/exp.hpp"
#include "subgraph_simple.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "openvino/op/subtract.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string Exp::getTestCaseName(testing::TestParamInfo<ov::test::snippets::ExpParams> obj) {
    ov::test::InputShape inputShapes0;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes0, type, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({inputShapes0.first}) << "_";
    result << "TS[0]=";
    for (const auto& shape : inputShapes0.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void Exp::SetUp() {
    ov::test::InputShape inputShape0;
    ov::element::Type type;
    std::tie(inputShape0, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({inputShape0});
    auto f = ov::test::snippets::ExpFunction(inputDynamicShapes);
    function = f.getOriginal();
    setInferenceType(type);
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

void ExpReciprocal::SetUp() {
    ov::test::InputShape inputShape0;
    ov::element::Type type;
    std::tie(inputShape0, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({inputShape0});
    auto f = ov::test::snippets::ExpReciprocalFunction(inputDynamicShapes);
    function = f.getOriginal();
    setInferenceType(type);
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}


TEST_P(Exp, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ExpReciprocal, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
