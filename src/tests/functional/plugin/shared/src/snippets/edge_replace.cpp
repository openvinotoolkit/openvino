// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/edge_replace.hpp"
#include "subgraph_simple.hpp"
#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string EdgeReplace::getTestCaseName(testing::TestParamInfo<ov::test::snippets::EdgeReplaceParams> obj) {
    ov::test::InputShape inputShapes0, inputShapes1, inputShapes2, inputShapes3, inputShapes4, inputShapes5, inputShapes6, inputShapes7;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes0, inputShapes1, inputShapes2, inputShapes3, inputShapes4, inputShapes5, inputShapes6, inputShapes7,
        type, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::partialShape2str({inputShapes0.first}) << "_";
    result << "TS[0]=";
    for (const auto& shape : inputShapes0.second) {
        result << "(" << CommonTestUtils::vec2str(shape) << ")_";
    }
    result << "IS[1]=" << CommonTestUtils::partialShape2str({inputShapes1.first}) << "_";
    result << "TS[1]=";
    for (const auto& shape : inputShapes1.second) {
        result << "(" << CommonTestUtils::vec2str(shape) << ")_";
    }
    result << "IS[2]=" << CommonTestUtils::partialShape2str({inputShapes2.first}) << "_";
    result << "TS[2]=";
    for (const auto& shape : inputShapes2.second) {
        result << "(" << CommonTestUtils::vec2str(shape) << ")_";
    }
    result << "IS[3]=" << CommonTestUtils::partialShape2str({inputShapes3.first}) << "_";
    result << "TS[3]=";
    for (const auto& shape : inputShapes3.second) {
        result << "(" << CommonTestUtils::vec2str(shape) << ")_";
    }
    result << "IS[4]=" << CommonTestUtils::partialShape2str({inputShapes4.first}) << "_";
    result << "TS[4]=";
    for (const auto& shape : inputShapes4.second) {
        result << "(" << CommonTestUtils::vec2str(shape) << ")_";
    }
    result << "IS[5]=" << CommonTestUtils::partialShape2str({inputShapes5.first}) << "_";
    result << "TS[5]=";
    for (const auto& shape : inputShapes5.second) {
        result << "(" << CommonTestUtils::vec2str(shape) << ")_";
    }
    result << "IS[6]=" << CommonTestUtils::partialShape2str({inputShapes6.first}) << "_";
    result << "TS[6]=";
    for (const auto& shape : inputShapes6.second) {
        result << "(" << CommonTestUtils::vec2str(shape) << ")_";
    }
    result << "IS[7]=" << CommonTestUtils::partialShape2str({inputShapes7.first}) << "_";
    result << "TS[7]=";
    for (const auto& shape : inputShapes7.second) {
        result << "(" << CommonTestUtils::vec2str(shape) << ")_";
    }
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EdgeReplace::SetUp() {
    ov::test::InputShape inputShape0, inputShape1, inputShape2, inputShape3, inputShape4, inputShape5, inputShape6, inputShape7;
    ov::element::Type type;
    std::tie(inputShape0, inputShape1, inputShape2, inputShape3, inputShape4, inputShape5, inputShape6, inputShape7,
        type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{inputShape0}, {inputShape1}, {inputShape2}, {inputShape3}, {inputShape4}, {inputShape5}, {inputShape6}, {inputShape7}});

    bool is_dynamic = inputShape0.first.is_dynamic();
    auto f = ov::test::snippets::EdgeReplaceFunction({is_dynamic ? inputShape0.first : inputShape0.second[0],
                                                      is_dynamic ? inputShape1.first : inputShape1.second[0],
                                                      is_dynamic ? inputShape2.first : inputShape2.second[0],
                                                      is_dynamic ? inputShape3.first : inputShape3.second[0],
                                                      is_dynamic ? inputShape4.first : inputShape4.second[0],
                                                      is_dynamic ? inputShape5.first : inputShape5.second[0],
                                                      is_dynamic ? inputShape6.first : inputShape6.second[0],
                                                      is_dynamic ? inputShape7.first : inputShape7.second[0]});
    function = f.getOriginal();
    setInferenceType(type);
}

TEST_P(EdgeReplace, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
