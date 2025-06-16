// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/lstm_cell.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "transformations/op_conversions/lstm_cell_decomposition.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_cell.hpp"

namespace ov {
namespace test {
using ov::test::utils::InputLayerType;

std::string LSTMCellTest::getTestCaseName(const testing::TestParamInfo<LSTMCellParams> &obj) {
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    InputLayerType WType;
    InputLayerType RType;
    InputLayerType BType;
    ov::element::Type model_type;
    std::string targetDevice;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, WType, RType, BType,
            model_type, targetDevice) = obj.param;
    std::vector<std::vector<size_t>> input_shapes = {
            {{batch, input_size}, {batch, hidden_size}, {batch, hidden_size}, {4 * hidden_size, input_size},
                    {4 * hidden_size, hidden_size}, {4 * hidden_size}},
    };
    std::ostringstream result;
    result << "decomposition" << should_decompose << "_";
    result << "batch=" << batch << "_";
    result << "hidden_size=" << hidden_size << "_";
    result << "input_size=" << input_size << "_";
    result << "IS=" << ov::test::utils::vec2str(input_shapes) << "_";
    result << "activations=" << ov::test::utils::vec2str(activations) << "_";
    result << "clip=" << clip << "_";
    result << "WType=" << WType << "_";
    result << "RType=" << RType << "_";
    result << "BType=" << BType << "_";
    result << "modelType=" << model_type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    return result.str();
}

void LSTMCellTest::SetUp() {
    bool should_decompose;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    InputLayerType WType;
    InputLayerType RType;
    InputLayerType BType;
    ov::element::Type model_type;
    std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, WType, RType, BType,
            model_type, targetDevice) = this->GetParam();

    std::vector<ov::Shape> input_shapes = {
        {batch, input_size},
        {batch, hidden_size},
        {batch, hidden_size},
        {4 * hidden_size, input_size},
        {4 * hidden_size, hidden_size},
        {4 * hidden_size}
    };

    std::vector<ov::Shape> param_shapes{input_shapes[0], input_shapes[1], input_shapes[2]};
    if (WType == InputLayerType::PARAMETER)
        param_shapes.push_back(input_shapes[3]);

    if (RType == InputLayerType::PARAMETER)
        param_shapes.push_back(input_shapes[4]);

    if (BType == InputLayerType::PARAMETER)
        param_shapes.push_back(input_shapes[5]);
    init_input_shapes(ov::test::static_shapes_to_test_representation(param_shapes));

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                               std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
                               std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};


    ov::NodeVector inputs{params[0], params[1], params[2]};
    if (WType == InputLayerType::PARAMETER) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[params.size()]);
        params.push_back(param);
        inputs.push_back(param);
    } else {
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, input_shapes[3]);
        auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
        inputs.push_back(constant);
    }

    if (RType == InputLayerType::PARAMETER) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[params.size()]);
        params.push_back(param);
        inputs.push_back(param);
    } else {
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, input_shapes[4]);
        auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
        inputs.push_back(constant);
    }

    if (BType == InputLayerType::PARAMETER) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[params.size()]);
        params.push_back(param);
        inputs.push_back(param);
    } else {
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, input_shapes[5]);
        auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
        inputs.push_back(constant);
    }

    auto lstm_cell = std::make_shared<ov::op::v4::LSTMCell>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                                            hidden_size, activations,
                                                            activations_alpha, activations_beta, clip);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lstm_cell->output(0)),
                                 std::make_shared<ov::op::v0::Result>(lstm_cell->output(1))};
    function = std::make_shared<ov::Model>(results, params, "lstm_cell");
    if (should_decompose) {
        ov::pass::Manager m;
        m.register_pass<ov::pass::LSTMCellDecomposition>();
        m.run_passes(function);
    }
}
}  // namespace test
}  // namespace ov
