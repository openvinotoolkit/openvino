// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/experimental_detectron_generate_proposals_single_image.hpp"

namespace ov {
namespace test {
std::string ExperimentalDetectronGenerateProposalsSingleImageLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronGenerateProposalsSingleImageTestParams>& obj) {
    std::vector<InputShape> shapes;
    ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes attributes;
    ElementType model_type;
    std::string device_name;
    std::tie(
        shapes,
        attributes.min_size,
        attributes.nms_threshold,
        attributes.post_nms_count,
        attributes.pre_nms_count,
        model_type,
        device_name) = obj.param;

    std::ostringstream result;
    using ov::test::operator<<;
    result << "im_info=" << shapes[0] << "_";
    result << "anchors=" << shapes[1] << "_";
    result << "deltas=" << shapes[2] << "_";
    result << "scores=" << shapes[3] << "_";

    result << "attributes={";
    result << "score_threshold=" << attributes.min_size << "_";
    result << "nms_threshold=" << attributes.nms_threshold << "_";
    result << "max_delta_log_wh=" << attributes.post_nms_count << "_";
    result << "num_classes=" << attributes.pre_nms_count;
    result << "}_";

    result << "netPRC=" << model_type << "_";
    result << "trgDev=" << device_name;
    return result.str();
}

void ExperimentalDetectronGenerateProposalsSingleImageLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes attributes;
    ElementType model_type;
    std::string targetName;
    std::tie(
        shapes,
        attributes.min_size,
        attributes.nms_threshold,
        attributes.post_nms_count,
        attributes.pre_nms_count,
        model_type,
        targetName) = this->GetParam();

    inType = outType = model_type;
    targetDevice = targetName;

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }
    auto experimental_detectron = std::make_shared<ov::op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(
        params[0], // im_info
        params[1], // anchors
        params[2], // deltas
        params[3], // scores
        attributes);
    function = std::make_shared<ov::Model>(
        ov::OutputVector{experimental_detectron->output(0), experimental_detectron->output(1)},
        params,
        "ExperimentalDetectronGenerateProposalsSingleImage");
}
} // namespace test
} // namespace ov
