// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/generate_proposals.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
namespace subgraph {

namespace {
std::ostream& operator <<(
        std::ostream& ss,
        const ov::op::v9::GenerateProposals::Attributes& attributes) {
    ss << "score_threshold=" << attributes.min_size << "_";
    ss << "nms_threshold=" << attributes.nms_threshold << "_";
    ss << "max_delta_log_wh=" << attributes.post_nms_count << "_";
    ss << "num_classes=" << attributes.pre_nms_count;
    return ss;
}
} // namespace

std::string GenerateProposalsLayerTest::getTestCaseName(
        const testing::TestParamInfo<GenerateProposalsTestParams>& obj) {
    std::vector<InputShape> inputShapes;
    ov::op::v9::GenerateProposals::Attributes attributes;
    std::pair<std::string, std::vector<ov::Tensor>> inputTensors;
    ElementType netPrecision;
    ElementType roiNumPrecision;
    std::string targetName;
    std::tie(
        inputShapes,
        attributes.min_size,
        attributes.nms_threshold,
        attributes.post_nms_count,
        attributes.pre_nms_count,
        inputTensors,
        netPrecision,
        roiNumPrecision,
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
    result << "roiNumPRC=" << roiNumPrecision << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void GenerateProposalsLayerTest::SetUp() {
    std::vector<InputShape> inputShapes;
    ov::op::v9::GenerateProposals::Attributes attributes;
    std::pair<std::string, std::vector<ov::Tensor>> inputTensors;
    ElementType netPrecision;
    ElementType roiNumPrecision;
    std::string targetName;
    std::tie(
        inputShapes,
        attributes.min_size,
        attributes.nms_threshold,
        attributes.post_nms_count,
        attributes.pre_nms_count,
        inputTensors,
        netPrecision,
        roiNumPrecision,
        targetName) = this->GetParam();

    inType = outType = netPrecision;
    targetDevice = targetName;

    init_input_shapes(inputShapes);

    auto params = ngraph::builder::makeDynamicParams(netPrecision, {inputDynamicShapes});
    auto paramsOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto generateProposals = std::make_shared<ov::op::v9::GenerateProposals>(
        params[0], // im_info
        params[1], // anchors
        params[2], // deltas
        params[3], // scores
        attributes,
        roiNumPrecision);
    function = std::make_shared<ov::Model>(
        ov::OutputVector{generateProposals->output(0),
                         generateProposals->output(1),
                         generateProposals->output(2)},
        "GenerateProposals");
}

void GenerateProposalsLayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    auto inputTensors = std::get<5>(GetParam());

    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (auto i = 0ul; i < funcInputs.size(); ++i) {
        if (targetInputStaticShapes[i] != inputTensors.second[i].get_shape()) {
            throw Exception("input shape is different from tensor shape");
        }

        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensors.second[i]});
    }
}

} // namespace subgraph
} // namespace test
} // namespace ov
