// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/experimental_detectron_prior_grid_generator.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/data_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace subgraph {

namespace {
std::ostream& operator <<(
        std::ostream& ss,
        const ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes& attributes) {
    ss << "flatten=" << attributes.flatten << "_";
    ss << "h=" << attributes.h << "_";
    ss << "w=" << attributes.w << "_";
    ss << "stride_x=" << attributes.stride_x << "_";
    ss << "stride_y=" << attributes.stride_y;
    return ss;
}
} // namespace

std::string ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronPriorGridGeneratorTestParams>& obj) {
    ExperimentalDetectronPriorGridGeneratorTestParam param;
    std::pair<std::string, std::vector<ov::Tensor>> inputTensors;
    ElementType netPrecision;
    std::string targetName;
    std::tie(param, inputTensors, netPrecision, targetName) = obj.param;

    std::ostringstream result;
    using ov::test::operator<<;
    result << "priors=" << param.inputShapes[0] << "_";
    result << "feature_map=" << param.inputShapes[1] << "_";
    result << "im_data=" << param.inputShapes[2] << "_";

    using ov::test::subgraph::operator<<;
    result << "attributes=" << param.attributes << "_";
    result << "priorValues=" << inputTensors.first << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExperimentalDetectronPriorGridGeneratorLayerTest::SetUp() {
    ExperimentalDetectronPriorGridGeneratorTestParam param;
    std::pair<std::string, std::vector<ov::Tensor>> inputTensors;
    ElementType netPrecision;
    std::string targetName;
    std::tie(param, inputTensors, netPrecision, targetName) = this->GetParam();

    inType = outType = netPrecision;
    targetDevice = targetName;

    init_input_shapes(param.inputShapes);

    auto params = ngraph::builder::makeDynamicParams(netPrecision, {inputDynamicShapes});
    auto paramsOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto experimentalDetectron = std::make_shared<op::v6::ExperimentalDetectronPriorGridGenerator>(
        params[0], // priors
        params[1], // feature_map
        params[2], // im_data
        param.attributes);
    function = std::make_shared<ov::Model>(
            ov::OutputVector{experimentalDetectron->output(0)},
            "ExperimentalDetectronPriorGridGenerator");
}

namespace {
template<typename T>
ov::runtime::Tensor generateTensorByShape(const Shape &shape) {
    return ov::test::utils::create_tensor<T>(
            ov::element::from<T>(),
            shape,
            std::vector<T>(0., shape_size(shape)));
}
}

void ExperimentalDetectronPriorGridGeneratorLayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    auto inputTensors = std::get<1>(GetParam());
    auto netPrecision = std::get<2>(GetParam());

    inputs.clear();
    const auto& funcInputs = function->inputs();

    auto i = 0ul;
    for (; i < inputTensors.second.size(); ++i) {
        if (targetInputStaticShapes[i] != inputTensors.second[i].get_shape()) {
            throw Exception("input shape is different from tensor shape");
        }

        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensors.second[i]});
    }
    for (auto j = i; j < funcInputs.size(); ++j) {
        ov::runtime::Tensor inputTensor = (netPrecision == element::f16)
                                          ? generateTensorByShape<ov::float16>(targetInputStaticShapes[j])
                                          : generateTensorByShape<float>(
                        targetInputStaticShapes[j]);

        inputs.insert({funcInputs[j].get_node_shared_ptr(), inputTensor});
    }
}

} // namespace subgraph
} // namespace test
} // namespace ov
