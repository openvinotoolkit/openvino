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
    const auto& [input_shapes, fake_quantize_intervals, num_nodes, num_subgraphs, targetDevice] = obj.param;

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
    const auto& [input_shapes, fake_quantize_intervals, _ref_num_nodes, _ref_num_subgraphs, _targetDevice] =
        this->GetParam();
    ref_num_nodes = _ref_num_nodes;
    ref_num_subgraphs = _ref_num_subgraphs;
    targetDevice = _targetDevice;
    init_input_shapes(input_shapes);

    function = PrecisionPropagationConvertionFunction(inputDynamicShapes, ov::element::f32, fake_quantize_intervals).getOriginal();
    setIgnoreCallbackMode();
}

TEST_P(PrecisionPropagationConvertion, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
