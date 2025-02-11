// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/group_normalization.hpp"
#include "subgraph_group_normalization.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string GroupNormalization::getTestCaseName(testing::TestParamInfo<ov::test::snippets::GroupNormalizationParams> obj) {
    InputShape inputShapes;
    size_t numGroup;
    float eps;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(inputShapes, numGroup, eps, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::partialShape2str({inputShapes.first}) << "_";
    result << "TS=";
    for (const auto& shape : inputShapes.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "numGroup=" << numGroup << "_";
    result << "epsilon=" << eps << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void GroupNormalization::SetUp() {
    InputShape inputShape;
    size_t numGroup;
    float eps;
    std::tie(inputShape, numGroup, eps, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();

    InputShape scaleShiftShape = ExtractScaleShiftShape(inputShape);

    init_input_shapes({inputShape, scaleShiftShape, scaleShiftShape});

    auto f = ov::test::snippets::GroupNormalizationFunction(inputDynamicShapes, numGroup, eps);
    function = f.getOriginal();

    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }

    abs_threshold = 1e-5;
}

InputShape GroupNormalization::ExtractScaleShiftShape(const InputShape& shape) {
    std::vector<ov::Shape> biasShape;
    std::transform(shape.second.cbegin(), shape.second.cend(), std::back_inserter(biasShape),
        [](const ov::Shape& s)->ov::Shape {
            OPENVINO_ASSERT(s.size() >= 2, "First input rank for group normalization op should be greater than 1");
            return {s[1]};
        });
    InputShape biasInputShape {
        shape.first.is_dynamic() ? ov::PartialShape{shape.first[1]} : shape.first,
        std::move(biasShape)
    };
    return biasInputShape;
}

TEST_P(GroupNormalization, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
