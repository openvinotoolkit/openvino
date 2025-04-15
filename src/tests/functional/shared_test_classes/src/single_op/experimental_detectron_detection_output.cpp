// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/experimental_detectron_detection_output.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
std::string ExperimentalDetectronDetectionOutputLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronDetectionOutputTestParams>& obj) {
    std::vector<ov::test::InputShape> shapes;
    ov::op::v6::ExperimentalDetectronDetectionOutput::Attributes attributes;
    ElementType model_type;
    std::string target_device;
    std::tie(
        shapes,
        attributes.score_threshold,
        attributes.nms_threshold,
        attributes.max_delta_log_wh,
        attributes.num_classes,
        attributes.post_nms_count,
        attributes.max_detections_per_image,
        attributes.class_agnostic_box_regression,
        attributes.deltas_weights,
        model_type,
        target_device) = obj.param;

    std::ostringstream result;

    using ov::test::operator<<;
    result << "input_rois=" << shapes[0] << "_";
    result << "input_deltas=" << shapes[1] << "_";
    result << "input_scores=" << shapes[2] << "_";
    result << "input_im_info=" << shapes[3] << "_";

    result << "attributes={";
    result << "score_threshold=" << attributes.score_threshold << "_";
    result << "nms_threshold=" << attributes.nms_threshold << "_";
    result << "max_delta_log_wh=" << attributes.max_delta_log_wh << "_";
    result << "num_classes=" << attributes.num_classes << "_";
    result << "post_nms_count=" << attributes.post_nms_count << "_";
    result << "max_detections_per_image=" << attributes.max_detections_per_image << "_";
    result << "class_agnostic_box_regression=" << (attributes.class_agnostic_box_regression ? "true" : "false") << "_";
    result << "deltas_weights=" << ov::test::utils::vec2str(attributes.deltas_weights);
    result << "}_";

    result << "modelType=" << model_type << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void ExperimentalDetectronDetectionOutputLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::op::v6::ExperimentalDetectronDetectionOutput::Attributes attributes;

    ElementType model_type;
    std::string targetName;
    std::tie(
        shapes,
        attributes.score_threshold,
        attributes.nms_threshold,
        attributes.max_delta_log_wh,
        attributes.num_classes,
        attributes.post_nms_count,
        attributes.max_detections_per_image,
        attributes.class_agnostic_box_regression,
        attributes.deltas_weights,
        model_type,
        targetName) = this->GetParam();

    if (model_type == element::f16)
        abs_threshold = 0.01;

    inType = outType = model_type;
    targetDevice = targetName;

    init_input_shapes(shapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }

    auto experimentalDetectron = std::make_shared<ov::op::v6::ExperimentalDetectronDetectionOutput>(
        params[0], // input_rois
        params[1], // input_deltas
        params[2], // input_scores
        params[3], // input_im_info
        attributes);
    function = std::make_shared<ov::Model>(
        ov::OutputVector{experimentalDetectron->output(0), experimentalDetectron->output(1)},
        "ExperimentalDetectronDetectionOutput");
}
} // namespace test
} // namespace ov
