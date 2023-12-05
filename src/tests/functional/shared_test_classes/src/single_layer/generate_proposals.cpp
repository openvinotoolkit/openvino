// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/generate_proposals.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/builders.hpp"
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
    ss << "post_nms_count=" << attributes.post_nms_count << "_";
    ss << "pre_nms_count=" << attributes.pre_nms_count;
    ss << "normalized=" << attributes.normalized;
    ss << "nms_eta=" << attributes.nms_eta;
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
        attributes.normalized,
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
        attributes.normalized,
        inputTensors,
        netPrecision,
        roiNumPrecision,
        targetName) = this->GetParam();

    inType = outType = netPrecision;
    targetDevice = targetName;
    if (targetDevice == ov::test::utils::DEVICE_GPU) {
        if (netPrecision == element::Type_t::f16) {
            abs_threshold = 0.2;
        } else {
            abs_threshold = 0.00009;
        }
    }

    init_input_shapes(inputShapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes)
        params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

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
    auto inputTensors = std::get<6>(GetParam());

    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (auto i = 0ul; i < funcInputs.size(); ++i) {
        if (targetInputStaticShapes[i] != inputTensors.second[i].get_shape()) {
            OPENVINO_THROW("input shape is different from tensor shape");
        }

        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensors.second[i]});
    }
}

void GenerateProposalsLayerTest::compare(const std::vector<ov::Tensor>& expected,
                                         const std::vector<ov::Tensor>& actual) {
    if (targetDevice != ov::test::utils::DEVICE_GPU) {
        SubgraphBaseTest::compare(expected, actual);
        return;
    }

    const auto outputsNum = expected.size();
    ASSERT_EQ(outputsNum, 3);
    ASSERT_EQ(outputsNum, actual.size());
    ASSERT_EQ(outputsNum, function->get_results().size());

    // actual outputs 0 (rois) and 1 (roi_scores) may be padded with zeros
    for (size_t i = 0; i < 2; ++i) {
        const auto expectedNumRois = expected[i].get_shape()[0];
        const auto actualNumRois = actual[i].get_shape()[0];
        ASSERT_LE(expectedNumRois, actualNumRois);

        const auto actualBuffer = static_cast<uint8_t*>(actual[i].data());
        const auto expectedBuffer = static_cast<uint8_t*>(expected[i].data());
        const auto outputSize = i == 0 ? 4 : 1;

        if (outType == element::Type_t::f32) {
            LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const float*>(expectedBuffer),
                                                       reinterpret_cast<const float*>(actualBuffer),
                                                       expectedNumRois * outputSize,
                                                       rel_threshold,
                                                       abs_threshold);
        } else {
            LayerTestsUtils::LayerTestsCommon::Compare(reinterpret_cast<const float16*>(expectedBuffer),
                                                       reinterpret_cast<const float16*>(actualBuffer),
                                                       expectedNumRois * outputSize,
                                                       rel_threshold,
                                                       abs_threshold);
        }

        if (expectedNumRois < actualNumRois) {
            if (outType == element::Type_t::f32) {
                const auto fBuffer = static_cast<const float*>(actual[i].data());
                for (size_t j = expectedNumRois * outputSize; j < actualNumRois * outputSize; ++j) {
                    ASSERT_TRUE(fBuffer[j] == 0.0f)
                        << "Expected 0.0, actual: " << fBuffer[j] << " at index: " << j << ", output: " << i;
                }
            } else {
                const float16 zero{0};
                const auto fBuffer = static_cast<const float16*>(actual[i].data());
                for (size_t j = expectedNumRois * outputSize; j < actualNumRois * outputSize; ++j) {
                    ASSERT_TRUE(fBuffer[j] == zero)
                        << "Expected 0.0, actual: " << fBuffer[j] << " at index: " << j << ", output: " << i;
                }
            }
        }
    }

    // output 2 - rois_num
    ov::test::utils::compare(expected[2], actual[2], abs_threshold, rel_threshold);
}
} // namespace subgraph
} // namespace test
} // namespace ov
