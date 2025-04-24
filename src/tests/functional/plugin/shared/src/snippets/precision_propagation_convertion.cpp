// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/precision_propagation_convertion.hpp"

#include "common_test_utils/common_utils.hpp"
#include "precision_propagation_convertion.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string PrecisionPropagationConvertion::getTestCaseName(testing::TestParamInfo<PrecisionPropagationParams> obj) {
    std::vector<InputShape> input_shapes;
    std::vector<float> fake_quantize_intervals;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, fake_quantize_intervals, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        result << "IS[" << i << "]=" << ov::test::utils::partialShape2str({input_shapes[i].first}) << "_";
        result << "TS[" << i << "}=";
        for (const auto& shape : input_shapes[i].second) {
            result << "(" << ov::test::utils::vec2str(shape) << ")_";
        }
    }
    for (size_t i = 0; i < fake_quantize_intervals.size(); ++i)
        result << "FQ[" << i << "]=" << fake_quantize_intervals[i] << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void PrecisionPropagationConvertion::SetUp() {
    std::vector<InputShape> input_shapes;
    std::vector<float> fake_quantize_intervals;
    std::tie(input_shapes, fake_quantize_intervals, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(input_shapes);

    function = PrecisionPropagationConvertionFunction(inputDynamicShapes, ov::element::f32, fake_quantize_intervals).getOriginal();
    if (!configuration.count("SNIPPETS_MODE")) {
        configuration.insert({"SNIPPETS_MODE", "IGNORE_CALLBACK"});
    }
}

TEST_P(PrecisionPropagationConvertion, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
