// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/experimental_detectron_prior_grid_generator.hpp"
#include "ngraph_functions/builders.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"

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
    ElementType netPrecision;
    std::string targetName;
    std::tie(param, netPrecision, targetName) = obj.param;

    std::ostringstream result;
    using ov::test::operator<<;
    result << "priors=" << param.inputShapes[0] << "_";
    result << "feature_map=" << param.inputShapes[1] << "_";
    result << "im_data=" << param.inputShapes[2] << "_";

    using ov::test::subgraph::operator<<;
    result << "attributes=" << param.attributes << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExperimentalDetectronPriorGridGeneratorLayerTest::SetUp() {
    ExperimentalDetectronPriorGridGeneratorTestParam param;
    ElementType netPrecision;
    std::string targetName;
    std::tie(param, netPrecision, targetName) = this->GetParam();

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
    function = std::make_shared<ov::Function>(
            ov::OutputVector{experimentalDetectron->output(0)},
            "ExperimentalDetectronPriorGridGenerator");
}
} // namespace subgraph
} // namespace test
} // namespace ov
