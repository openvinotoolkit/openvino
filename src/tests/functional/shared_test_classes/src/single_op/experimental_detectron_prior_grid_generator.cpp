// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/experimental_detectron_prior_grid_generator.hpp"

namespace ov {
namespace test {

std::string ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronPriorGridGeneratorTestParams>& obj) {
    std::vector<InputShape> shapes;
    ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes attributes;
    std::pair<std::string, std::vector<ov::Tensor>> inputTensors;
    ElementType model_type;
    std::string targetName;
    std::tie(shapes, attributes, model_type, targetName) = obj.param;

    std::ostringstream result;
    using ov::test::operator<<;
    result << "priors=" << shapes[0] << "_";
    result << "feature_map=" << shapes[1] << "_";
    result << "im_data=" << shapes[2] << "_";

    result << "attributes=";
    result << "flatten=" << attributes.flatten << "_";
    result << "h=" << attributes.h << "_";
    result << "w=" << attributes.w << "_";
    result << "stride_x=" << attributes.stride_x << "_";
    result << "stride_y=" << attributes.stride_y;
    result << "_";

    result << "priorValues=" << inputTensors.first << "_";
    result << "netPRC=" << model_type << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExperimentalDetectronPriorGridGeneratorLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes attributes;
    ElementType model_type;
    std::string targetName;
    std::tie(shapes, attributes, model_type, targetName) = this->GetParam();

    inType = outType = model_type;
    targetDevice = targetName;

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }
    auto experimentalDetectron = std::make_shared<op::v6::ExperimentalDetectronPriorGridGenerator>(
        params[0], // priors
        params[1], // feature_map
        params[2], // im_data
        attributes);
    function = std::make_shared<ov::Model>(
            ov::OutputVector{experimentalDetectron->output(0)},
            params,
            "ExperimentalDetectronPriorGridGenerator");
}

} // namespace test
} // namespace ov
