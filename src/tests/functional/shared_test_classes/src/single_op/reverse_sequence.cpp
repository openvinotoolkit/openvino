// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/reverse_sequence.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
std::string ReverseSequenceLayerTest::getTestCaseName(const testing::TestParamInfo<ReverseSequenceParamsTuple> &obj) {
    int64_t batch_axis_idx;
    int64_t seq_axis_idx;
    ov::element::Type model_type;
    std::string target_device;
    std::vector<size_t> input_shape;
    std::vector<size_t> second_input_shape;
    ov::test::utils::InputLayerType secondary_input_type;

    std::tie(batch_axis_idx, seq_axis_idx, input_shape, second_input_shape, secondary_input_type, model_type, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "seqLengthsShape" << ov::test::utils::vec2str(second_input_shape) << "_";
    result << "secondaryInputType=" << secondary_input_type << "_";
    result << "batchAxis=" << batch_axis_idx << "_";
    result << "seqAxis=" << seq_axis_idx << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void ReverseSequenceLayerTest::SetUp() {
    ov::element::Type model_type;
    int64_t batch_axis_idx;
    int64_t seq_axis_idx;
    std::vector<size_t> input_shape;
    std::vector<size_t> second_input_shape;
    ov::test::utils::InputLayerType secondary_input_type;

    std::tie(batch_axis_idx, seq_axis_idx, input_shape, second_input_shape, secondary_input_type, model_type, targetDevice) = GetParam();

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape))};
    auto second_data_type = ov::element::i32; //according to the specification
    std::shared_ptr<ov::Node> secondary_input;
    if (ov::test::utils::InputLayerType::CONSTANT == secondary_input_type) {
        auto tensor = ov::test::utils::create_and_fill_tensor(second_data_type, second_input_shape);
        secondary_input = std::make_shared<ov::op::v0::Constant>(tensor);
    } else if (ov::test::utils::InputLayerType::PARAMETER == secondary_input_type) {
        secondary_input = std::make_shared<ov::op::v0::Parameter>(second_data_type, ov::Shape(second_input_shape));
        params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(secondary_input));
    } else {
        throw std::runtime_error("Unsupported input type");
    }

    auto reverse = std::make_shared<ov::op::v0::ReverseSequence>(params[0], secondary_input, batch_axis_idx, seq_axis_idx);
    function = std::make_shared<ov::Model>(reverse->outputs(), params, "ReverseSequence");
}
}  // namespace test
}  // namespace ov
