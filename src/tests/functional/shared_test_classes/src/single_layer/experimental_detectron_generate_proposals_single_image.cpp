// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/experimental_detectron_generate_proposals_single_image.hpp"
#include "ov_models/builders.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace subgraph {

namespace {
std::ostream& operator <<(
        std::ostream& ss,
        const ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes& attributes) {
    ss << "score_threshold=" << attributes.min_size << "_";
    ss << "nms_threshold=" << attributes.nms_threshold << "_";
    ss << "max_delta_log_wh=" << attributes.post_nms_count << "_";
    ss << "num_classes=" << attributes.pre_nms_count;
    return ss;
}
} // namespace

std::string ExperimentalDetectronGenerateProposalsSingleImageLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronGenerateProposalsSingleImageTestParams>& obj) {
    std::vector<InputShape> inputShapes;
    ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes attributes;
    std::pair<std::string, std::vector<ov::Tensor>> inputTensors;
    ElementType netPrecision;
    std::string targetName;
    std::tie(
        inputShapes,
        attributes.min_size,
        attributes.nms_threshold,
        attributes.post_nms_count,
        attributes.pre_nms_count,
        inputTensors,
        netPrecision,
        targetName) = obj.param;

    std::ostringstream result;
    using ov::test::operator<<;
    result << "im_info=" << inputShapes[0] << "_";
    result << "anchors=" << inputShapes[1] << "_";
    result << "deltas=" << inputShapes[2] << "_";
    result << "scores=" << inputShapes[3] << "_";

    using ov::test::subgraph::operator<<;
    result << "attributes={" << attributes << "}_";
    result << "inputTensors=" << inputTensors.first << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExperimentalDetectronGenerateProposalsSingleImageLayerTest::SetUp() {
    std::vector<InputShape> inputShapes;
    ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes attributes;
    std::pair<std::string, std::vector<ov::Tensor>> inputTensors;
    ElementType netPrecision;
    std::string targetName;
    std::tie(
        inputShapes,
        attributes.min_size,
        attributes.nms_threshold,
        attributes.post_nms_count,
        attributes.pre_nms_count,
        inputTensors,
        netPrecision,
        targetName) = this->GetParam();

    inType = outType = netPrecision;
    targetDevice = targetName;

    init_input_shapes(inputShapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
    }
    auto paramsOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto experimentalDetectron = std::make_shared<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(
        params[0], // im_info
        params[1], // anchors
        params[2], // deltas
        params[3], // scores
        attributes);
    function = std::make_shared<ov::Model>(
        ov::OutputVector{experimentalDetectron->output(0), experimentalDetectron->output(1)},
        "ExperimentalDetectronGenerateProposalsSingleImage");
}

void ExperimentalDetectronGenerateProposalsSingleImageLayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    auto inputTensors = std::get<5>(GetParam());

    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (auto i = 0ul; i < funcInputs.size(); ++i) {
        if (targetInputStaticShapes[i] != inputTensors.second[i].get_shape()) {
            OPENVINO_THROW("input shape is different from tensor shape");
        }

        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensors.second[i]});
    }
}

} // namespace subgraph
} // namespace test
} // namespace ov
