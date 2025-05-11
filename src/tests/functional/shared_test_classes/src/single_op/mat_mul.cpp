// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/mat_mul.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace test {
using ov::test::utils::InputLayerType;

std::string MatMulLayerTest::getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj) {
    std::vector<InputShape> shapes;
    std::pair<bool, bool> transpose;
    ov::element::Type model_type;
    InputLayerType secondary_input_type;
    std::string target_device;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes, transpose, model_type, secondary_input_type, target_device, additional_config) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "transpose_a=" << transpose.first << "_";
    result << "transpose_b=" << transpose.second << "_";
    result << "secondary_input_type=" << secondary_input_type << "_";
    result << "modelType=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    result << "config=(";
    for (const auto& configEntry : additional_config) {
        result << configEntry.first << ", " << configEntry.second << ";";
    }
    result << ")";
    return result.str();
}

void MatMulLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    std::pair<bool, bool> transpose;
    ov::element::Type model_type;
    InputLayerType secondary_input_type;
    std::map<std::string, std::string> additional_config;
    std::tie(shapes, transpose, model_type, secondary_input_type, targetDevice, additional_config) = this->GetParam();
    init_input_shapes(shapes);
    configuration.insert(additional_config.begin(), additional_config.end());

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};
    ov::NodeVector inputs {params[0]};

    if (InputLayerType::PARAMETER == secondary_input_type) {
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
        params.push_back(param);
        inputs.push_back(param);
    } else {
        auto tensor = ov::test::utils::create_and_fill_tensor(model_type, targetStaticShapes[0][1]);
        auto constant = std::make_shared<ov::op::v0::Constant>(tensor);
        inputs.push_back(constant);
    }
    auto mat_mul = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[1], transpose.first, transpose.second);

    auto result = std::make_shared<ov::op::v0::Result>(mat_mul);

    function = std::make_shared<ov::Model>(result, params, "MatMul");
}
}  // namespace test
}  // namespace ov
