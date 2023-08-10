// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/precision_propagation_convertion.hpp"

#include "common_test_utils/common_utils.hpp"
#include "precision_propagation_convertion_function.hpp"

namespace ov {
namespace test {
namespace snippets {

std::string PrecisionPropagationConvertion::getTestCaseName(testing::TestParamInfo<PrecisionPropagationParams> obj) {
    std::vector<ov::PartialShape> input_shapes;
    std::vector<float> fake_quantize_intervals;
    std::string targetDevice;
    size_t num_nodes, num_subgraphs;
    std::tie(input_shapes, fake_quantize_intervals, num_nodes, num_subgraphs, targetDevice) = obj.param;

    std::ostringstream result;
    for (size_t i = 0; i < input_shapes.size(); ++i)
        result << "IS[" << i << "]=" << input_shapes[i] << "_";
    for (size_t i = 0; i < fake_quantize_intervals.size(); ++i)
        result << "FQ[" << i << "]=" << fake_quantize_intervals[i] << "_";
    result << "#N=" << num_nodes << "_";
    result << "#S=" << num_subgraphs << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void PrecisionPropagationConvertion::SetUp() {
    std::vector<ov::PartialShape> input_shapes;
    std::vector<float> fake_quantize_intervals;
    std::tie(input_shapes, fake_quantize_intervals, ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
    init_input_shapes(static_partial_shapes_to_test_representation(input_shapes));

    function = PrecisionPropagationConvertionFunction(input_shapes, ov::element::f32, fake_quantize_intervals).getOriginal();
}

TEST_P(PrecisionPropagationConvertion, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov
