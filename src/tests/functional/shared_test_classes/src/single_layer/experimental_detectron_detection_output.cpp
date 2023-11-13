// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/experimental_detectron_detection_output.hpp"
#include "ov_models/builders.hpp"
#include "common_test_utils/data_utils.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace subgraph {

namespace {
    std::ostream& operator <<(std::ostream& ss, const ngraph::opset6::ExperimentalDetectronDetectionOutput::Attributes& attributes) {
    ss << "score_threshold=" << attributes.score_threshold << "_";
    ss << "nms_threshold=" << attributes.nms_threshold << "_";
    ss << "max_delta_log_wh=" << attributes.max_delta_log_wh << "_";
    ss << "num_classes=" << attributes.num_classes << "_";
    ss << "post_nms_count=" << attributes.post_nms_count << "_";
    ss << "max_detections_per_image=" << attributes.max_detections_per_image << "_";
    ss << "class_agnostic_box_regression=" << (attributes.class_agnostic_box_regression ? "true" : "false") << "_";
    ss << "deltas_weights=" << ov::test::utils::vec2str(attributes.deltas_weights);
    return ss;
}
} // namespace

std::string ExperimentalDetectronDetectionOutputLayerTest::getTestCaseName(
        const testing::TestParamInfo<ExperimentalDetectronDetectionOutputTestParams>& obj) {
    std::vector<ov::test::InputShape> inputShapes;
    ngraph::opset6::ExperimentalDetectronDetectionOutput::Attributes attributes;
    ElementType netPrecision;
    std::string targetName;
    std::tie(
        inputShapes,
        attributes.score_threshold,
        attributes.nms_threshold,
        attributes.max_delta_log_wh,
        attributes.num_classes,
        attributes.post_nms_count,
        attributes.max_detections_per_image,
        attributes.class_agnostic_box_regression,
        attributes.deltas_weights,
        netPrecision,
        targetName) = obj.param;

    std::ostringstream result;

    using ov::test::operator<<;
    result << "input_rois=" << inputShapes[0] << "_";
    result << "input_deltas=" << inputShapes[1] << "_";
    result << "input_scores=" << inputShapes[2] << "_";
    result << "input_im_info=" << inputShapes[3] << "_";

    using ov::test::subgraph::operator<<;
    result << "attributes={" << attributes << "}_";
    result << "netPRC=" << netPrecision << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ExperimentalDetectronDetectionOutputLayerTest::SetUp() {
    std::vector<InputShape> inputShapes;
    ngraph::opset6::ExperimentalDetectronDetectionOutput::Attributes attributes;

    ElementType netPrecision;
    std::string targetName;
    std::tie(
        inputShapes,
        attributes.score_threshold,
        attributes.nms_threshold,
        attributes.max_delta_log_wh,
        attributes.num_classes,
        attributes.post_nms_count,
        attributes.max_detections_per_image,
        attributes.class_agnostic_box_regression,
        attributes.deltas_weights,
        netPrecision,
        targetName) = this->GetParam();

    if (netPrecision == element::f16)
        abs_threshold = 0.01;

    inType = outType = netPrecision;
    targetDevice = targetName;

    init_input_shapes(inputShapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes)
        params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));

    auto experimentalDetectron = std::make_shared<ngraph::opset6::ExperimentalDetectronDetectionOutput>(
        params[0], // input_rois
        params[1], // input_deltas
        params[2], // input_scores
        params[3], // input_im_info
        attributes);
    function = std::make_shared<ov::Model>(
        ov::OutputVector{experimentalDetectron->output(0), experimentalDetectron->output(1)},
        "ExperimentalDetectronDetectionOutput");
}

namespace {

template <typename T>
std::vector<T> getValues(const std::vector<float>& values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template <typename T>
std::vector<ov::Tensor> generateInputTensors() {
    const auto netPrecision = ov::element::from<T>();
    std::vector<ov::Tensor> inputTensors = {
        // 16 x 4 = 64
        ov::test::utils::create_tensor<T>(
            netPrecision,
            Shape{16, 4},
            getValues<T>({
            1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,  4.0f,  1.0f, 8.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        })),
        // 16 x 8
        ov::test::utils::create_tensor<T>(
            netPrecision,
            Shape{16, 8},
            getValues<T>({
            5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        })),
        // 16 x 2 = 32
        ov::test::utils::create_tensor<T>(
            netPrecision,
            Shape{16, 2},
            getValues<T>({
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        })),
        // 1 x 3 = 3
        ov::test::utils::create_tensor<T>(netPrecision, Shape{1, 3}, getValues<T>({1.0f, 1.0f, 1.0f}))};

    return inputTensors;
}
}  // namespace

void ExperimentalDetectronDetectionOutputLayerTest::generate_inputs(
    const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    const auto netPrecision = std::get<9>(GetParam());

    const std::vector<ov::Tensor> inputTensors =
        (netPrecision == element::f16) ? generateInputTensors<ov::float16>() : generateInputTensors<float>();

    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (auto i = 0ul; i < funcInputs.size(); ++i) {
        if (targetInputStaticShapes[i] != inputTensors[i].get_shape()) {
            OPENVINO_THROW("input shape is different from tensor shape");
        }

        inputs.insert({funcInputs[i].get_node_shared_ptr(), inputTensors[i]});
    }
}

} // namespace subgraph
} // namespace test
} // namespace ov
