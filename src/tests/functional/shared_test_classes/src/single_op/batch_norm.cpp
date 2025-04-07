// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/batch_norm.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
std::string BatchNormLayerTest::getTestCaseName(const testing::TestParamInfo<BatchNormLayerTestParams>& obj) {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    double epsilon;
    std::string target_device;
    std::tie(epsilon, model_type, shapes, target_device) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    result << "inT=" << model_type.get_type_name() << "_";
    result << "epsilon=" << epsilon << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void BatchNormLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    double epsilon;
    std::tie(epsilon, model_type, shapes, targetDevice) = this->GetParam();
    init_input_shapes(shapes);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};

    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 1;

    auto constant_shape = ov::Shape{params[0]->get_shape().at(1)};
    auto gamma_tensor = ov::test::utils::create_and_fill_tensor(model_type, constant_shape, in_data);
    auto gamma = std::make_shared<ov::op::v0::Constant>(gamma_tensor);

    auto beta_tensor = ov::test::utils::create_and_fill_tensor(model_type, constant_shape, in_data);
    auto beta = std::make_shared<ov::op::v0::Constant>(beta_tensor);

    auto mean_tensor = ov::test::utils::create_and_fill_tensor(model_type, constant_shape, in_data);
    auto mean = std::make_shared<ov::op::v0::Constant>(mean_tensor);

    // Fill the vector for variance with positive values
    in_data.range = 10;
    auto variance_tensor = ov::test::utils::create_and_fill_tensor(model_type, constant_shape, in_data);
    auto variance = std::make_shared<ov::op::v0::Constant>(variance_tensor);
    auto batch_norm = std::make_shared<ov::op::v5::BatchNormInference>(params[0], gamma, beta, mean, variance, epsilon);

    function = std::make_shared<ov::Model>(ov::OutputVector{batch_norm}, params, "BatchNormInference");
}
}  // namespace test
}  // namespace ov
