// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/max_num_params_eltwise.hpp"
#include "subgraph_simple.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string MaxNumParamsEltwiseSinh::getTestCaseName(testing::TestParamInfo<ov::test::snippets::MaxNumParamsEltwiseParams> obj) {
    ov::Shape inputShapes;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MaxNumParamsEltwiseSinh::SetUp() {
    ov::Shape inputShape;
    std::tie(inputShape, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    std::vector<ov::Shape> expandedShapes(10, inputShape);
    std::vector<InputShape> input_shapes;
    for (const auto& s : expandedShapes) {
        input_shapes.emplace_back(InputShape {{}, {s, }});
    }

    init_input_shapes(input_shapes);

    auto f = ov::test::snippets::EltwiseMaxNumParamsSinhFunction(expandedShapes);
    function = f.getOriginal();
}

TEST_P(MaxNumParamsEltwiseSinh, CompareWithRefImpl) {
    run();
    ov::pass::Serialize("model.xml", "model.bin").run_on_model(function);
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
