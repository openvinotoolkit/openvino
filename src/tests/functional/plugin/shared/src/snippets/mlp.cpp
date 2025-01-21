// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_matmul.hpp"

namespace ov {
namespace test {
namespace snippets {


std::string MLP::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MLPParams> obj) {
    std::vector<InputShape> input_shapes;
    ov::element::Type elem_type;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, elem_type, num_nodes, num_subgraphs, targetDevice) = obj.param;
    std::ostringstream result;
    OPENVINO_ASSERT(input_shapes.size() == 1, "MLP test supports only one input");
    result << "IS[" << 0 << "]=" << input_shapes[0] << "_";
    result << "T[" << 0 <<"]=" << elem_type << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MLP::SetUp() {
    std::vector<InputShape> input_shapes;
    ov::element::Type elem_type;
    std::tie(input_shapes, elem_type, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    auto mlp = std::make_shared<MLPFunction>(inputDynamicShapes, elem_type);

    function = mlp->getOriginal();
    inType = outType = elem_type;
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

TEST_P(MLP, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
