// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/rnn_cell_decomposition.hpp"
#include "shared_test_classes/single_op/rnn_cell.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/rnn_cell.hpp"

namespace ov {
namespace test {

using utils::InputLayerType;

std::string RNNCellTest::getTestCaseName(const testing::TestParamInfo<RNNCellParams> &obj) {
    const auto& [should_decompose,
                 batch,
                 hidden_size,
                 input_size,
                 activations,
                 clip,
                 WType,
                 RType,
                 BType,
                 model_type,
                 target_device] = obj.param;
    std::vector<std::vector<size_t>> input_shapes = {{batch, input_size}, {batch, hidden_size},
                                     {hidden_size, input_size}, {hidden_size, hidden_size}, {hidden_size}};
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
    result << "modelType=" << model_type.to_string() << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void RNNCellTest::SetUp() {
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;

    const auto& [should_decompose,
                 batch,
                 hidden_size,
                 input_size,
                 activations,
                 clip,
                 WType,
                 RType,
                 BType,
                 model_type,
                 _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    std::vector<std::vector<size_t>> input_shapes = {{batch, input_size}, {batch, hidden_size},
                                                    {hidden_size, input_size}, {hidden_size, hidden_size}, {hidden_size}};
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shapes[0])),
                               std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shapes[1]))};
    std::vector<ov::Shape> WRB = {input_shapes[2], input_shapes[3], input_shapes[4]};

    std::shared_ptr<ov::Node> W;
    if (WType == InputLayerType::PARAMETER) {
        const auto param = std::make_shared<ov::op::v0::Parameter>(model_type, WRB[0]);
        W = param;
        params.push_back(param);
    } else {
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, WRB[0]);
        W = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    std::shared_ptr<ov::Node> R;
    if (RType == InputLayerType::PARAMETER) {
        const auto param = std::make_shared<ov::op::v0::Parameter>(model_type, WRB[1]);
        R = param;
        params.push_back(param);
    } else {
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, WRB[1]);
        R = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    std::shared_ptr<ov::Node> B;
    if (BType == InputLayerType::PARAMETER) {
        const auto param = std::make_shared<ov::op::v0::Parameter>(model_type, WRB[2]);
        B = param;
        params.push_back(param);
    } else {
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, WRB[2]);
        B = std::make_shared<ov::op::v0::Constant>(tensor);
    }

    auto rnn_cell = std::make_shared<ov::op::v0::RNNCell>(params[0], params[1], W, R, B, hidden_size, activations,
                                                          activations_alpha, activations_beta, clip);
    function = std::make_shared<ov::Model>(rnn_cell->outputs(), params, "rnn_cell");
    if (should_decompose) {
        ov::pass::Manager m;
        m.register_pass<ov::pass::RNNCellDecomposition>();
        m.run_passes(function);
    }
}
}  // namespace test
}  // namespace ov
