// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/edge_replace.hpp"
#include "subgraph_simple.hpp"
#include "ov_models/builders.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string EdgeReplace::getTestCaseName(testing::TestParamInfo<ov::test::snippets::EdgeReplaceParams> obj) {
    ov::PartialShape inputShape;
    ov::element::Type type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShape, type, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({inputShape}) << "_";
    result << "T=" << type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void EdgeReplace::SetUp() {
    ov::PartialShape inputShape;
    ov::element::Type type;
    std::tie(inputShape, type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes({{{inputShape}, {inputShape.get_shape()}}});

    auto f = ov::test::snippets::EdgeReplaceFunction({inputShape});
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
