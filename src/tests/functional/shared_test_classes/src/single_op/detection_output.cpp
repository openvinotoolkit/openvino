// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/detection_output.hpp"

namespace ov {
namespace test {
std::ostream& operator <<(std::ostream& result, const Attributes& attrs) {
    result << "Classes=" << attrs.num_classes << "_";
    result << "backgrId=" << attrs.background_label_id << "_";
    result << "topK="  << attrs.top_k << "_";
    result << "varEnc=" << attrs.variance_encoded_in_target << "_";
    result << "keepTopK=" << ov::test::utils::vec2str(attrs.keep_top_k) << "_";
    result << "codeType=" << attrs.code_type << "_";
    result << "shareLoc=" << attrs.share_location << "_";
    result << "nmsThr=" << attrs.nms_threshold << "_";
    result << "confThr=" << attrs.confidence_threshold << "_";
    result << "clipAfterNms=" << attrs.clip_after_nms << "_";
    result << "clipBeforeNms=" << attrs.clip_before_nms << "_";
    result << "decrId=" << attrs.decrease_label_id << "_";
    result << "norm=" << attrs.normalized << "_";
    result << "inH=" << attrs.input_height << "_";
    result << "inW=" << attrs.input_width << "_";
    result << "OS=" << attrs.objectness_score << "_";
    return result;
}

std::string DetectionOutputLayerTest::getTestCaseName(const testing::TestParamInfo<DetectionOutputParams>& obj) {
    DetectionOutputAttributes common_attrs;
    ParamsWhichSizeDepends specific_attrs;
    Attributes attrs;
    size_t batch;
    std::string targetDevice;
    std::tie(common_attrs, specific_attrs, batch, attrs.objectness_score, targetDevice) = obj.param;

    std::tie(attrs.num_classes, attrs.background_label_id, attrs.top_k, attrs.keep_top_k, attrs.code_type, attrs.nms_threshold, attrs.confidence_threshold,
             attrs.clip_after_nms, attrs.clip_before_nms, attrs.decrease_label_id) = common_attrs;

    const size_t numInputs = 5;
    std::vector<ov::Shape> input_shapes(numInputs);
    std::tie(attrs.variance_encoded_in_target, attrs.share_location, attrs.normalized, attrs.input_height, attrs.input_width,
             input_shapes[idxLocation], input_shapes[idxConfidence], input_shapes[idxPriors], input_shapes[idxArmConfidence],
             input_shapes[idxArmLocation]) = specific_attrs;

    if (input_shapes[idxArmConfidence].empty()) {
        input_shapes.resize(3);
    }

    for (size_t i = 0; i < input_shapes.size(); i++) {
        input_shapes[i][0] = batch;
    }

    std::ostringstream result;
    result << "IS = { ";
    result << "LOC=" << ov::test::utils::vec2str(input_shapes[0]) << "_";
    result << "CONF=" << ov::test::utils::vec2str(input_shapes[1]) << "_";
    result << "PRIOR=" << ov::test::utils::vec2str(input_shapes[2]);
    std::string armConf, armLoc;
    if (input_shapes.size() > 3) {
        armConf = "_ARM_CONF=" + ov::test::utils::vec2str(input_shapes[3]) + "_";
        armLoc = "ARM_LOC=" + ov::test::utils::vec2str(input_shapes[4]);
    }
    result << armConf;
    result << armLoc << " }_";

    result << attrs;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

void DetectionOutputLayerTest::SetUp() {
    DetectionOutputAttributes common_attrs;
    ParamsWhichSizeDepends specific_attrs;
    size_t batch;
    Attributes attrs;
    std::tie(common_attrs, specific_attrs, batch, attrs.objectness_score, targetDevice) = this->GetParam();

    std::tie(attrs.num_classes, attrs.background_label_id, attrs.top_k, attrs.keep_top_k, attrs.code_type, attrs.nms_threshold, attrs.confidence_threshold,
             attrs.clip_after_nms, attrs.clip_before_nms, attrs.decrease_label_id) = common_attrs;

    std::vector<ov::Shape> input_shapes;
    input_shapes.resize(numInputs);
    std::tie(attrs.variance_encoded_in_target, attrs.share_location, attrs.normalized, attrs.input_height, attrs.input_width,
             input_shapes[idxLocation], input_shapes[idxConfidence], input_shapes[idxPriors], input_shapes[idxArmConfidence],
             input_shapes[idxArmLocation]) = specific_attrs;

    if (input_shapes[idxArmConfidence].empty()) {
        input_shapes.resize(3);
    }

    for (size_t i = 0; i < input_shapes.size(); i++) {
        input_shapes[i][0] = batch;
    }
    init_input_shapes(static_shapes_to_test_representation(input_shapes));

    ov::ParameterVector params;
    for (const auto& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
    }

    std::shared_ptr<ov::op::v0::DetectionOutput> det_out;
    if (params.size() == 3)
        det_out = std::make_shared<ov::op::v0::DetectionOutput>(params[0], params[1], params[2], attrs);
    else if (params.size() == 5)
        det_out = std::make_shared<ov::op::v0::DetectionOutput>(params[0],
                                                                params[1],
                                                                params[2],
                                                                params[3],
                                                                params[4],
                                                                attrs);
    auto result = std::make_shared<ov::op::v0::Result>(det_out);
    function = std::make_shared<ov::Model>(result, params, "DetectionOutput");
}
}  // namespace test
}  // namespace ov
