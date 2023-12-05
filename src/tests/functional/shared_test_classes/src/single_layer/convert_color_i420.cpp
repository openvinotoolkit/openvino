// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convert_color_i420.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/i420_to_bgr.hpp"

namespace LayerTestsDefinitions {

std::string ConvertColorI420LayerTest::getTestCaseName(const testing::TestParamInfo<ConvertColorI420ParamsTuple> &obj) {
    ov::Shape inputShape;
    ov::element::Type type;
    bool conversion, singlePlane;
    std::string targetName;
    std::tie(inputShape, type, conversion, singlePlane, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "netPRC=" << type.c_type_string() << "_";
    result << "convRGB=" << conversion << "_";
    result << "singlePlane=" << singlePlane << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void ConvertColorI420LayerTest::SetUp() {
    ov::Shape inputShape;
    ov::element::Type ngPrc;
    bool conversionToRGB, singlePlane;
    abs_threshold = 1.0f; // I420 conversion can use various algorithms, thus some absolute deviation is allowed
    threshold = 1.f; // Ignore relative comparison for I420 convert (allow 100% relative deviation)
    std::tie(inputShape, ngPrc, conversionToRGB, singlePlane, targetDevice) = GetParam();
    if (singlePlane) {
        inputShape[1] = inputShape[1] * 3 / 2;
        auto param = std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShape);
        std::shared_ptr<ov::Node> convert_color;
        if (conversionToRGB) {
            convert_color = std::make_shared<ov::op::v8::I420toRGB>(param);
        } else {
            convert_color = std::make_shared<ov::op::v8::I420toBGR>(param);
        }
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                      ov::ParameterVector{param}, "ConvertColorI420");
    } else {
        auto uvShape = ov::Shape{inputShape[0], inputShape[1] / 2, inputShape[2] / 2, 1};
        auto param_y = std::make_shared<ov::op::v0::Parameter>(ngPrc, inputShape);
        auto param_u = std::make_shared<ov::op::v0::Parameter>(ngPrc, uvShape);
        auto param_v = std::make_shared<ov::op::v0::Parameter>(ngPrc, uvShape);
        std::shared_ptr<ov::Node> convert_color;
        if (conversionToRGB) {
            convert_color = std::make_shared<ov::op::v8::I420toRGB>(param_y, param_u, param_v);
        } else {
            convert_color = std::make_shared<ov::op::v8::I420toBGR>(param_y, param_u, param_v);
        }
        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(convert_color),
                                                      ov::ParameterVector{param_y, param_u, param_v},
                                                      "ConvertColorI420");
    }
}

// -------- Accuracy test (R/G/B combinations) --------

void ConvertColorI420AccuracyTest::GenerateInputs() {
    inputs.clear();
    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    const auto& functionParams = function->get_parameters();
    for (const auto& param : functionParams) {
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
        InferenceEngine::InputInfo::CPtr info = infoIt->second;
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info->getTensorDesc());
        blob->allocate();
        size_t full_height = param->get_shape()[1];
        size_t full_width = param->get_shape()[2];
        int b_dim = static_cast<int>(full_height * 2 / (3 * full_width));
        ASSERT_GT(b_dim, 1) << "Image height is invalid for I420 Accuracy test";
        ASSERT_EQ(255 % (b_dim - 1), 0) << "Image height is invalid for I420 Accuracy test";
        int b_step = 255 / (b_dim - 1);
        auto input_image = I420TestUtils::color_test_image(full_width, full_width, b_step);
        auto* rawBlobDataPtr = blob->buffer().as<uint8_t*>();
        for (size_t j = 0; j < input_image.size(); ++j) {
            rawBlobDataPtr[j] = input_image[j];
        }

        inputs.push_back(blob);
    }
}

void ConvertColorI420AccuracyTest::Validate() {
    ConvertColorI420LayerTest::Validate();

    ASSERT_FALSE(expected_output.empty());
    ASSERT_TRUE(actual_output);
    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual_output);
    const auto lockedMemory = memory->wmap();
    const auto* actualBuffer = lockedMemory.as<const float*>();

    // Allow less than 2% of deviations with 1 color step. 2% is experimental value
    // For different calculation methods - 1.4% deviation is observed
    I420TestUtils::ValidateColors(expected_output.data(), actualBuffer, expected_output.size(), 0.02);
}

std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> ConvertColorI420AccuracyTest::CalculateRefs() {
    auto refs = ConvertColorI420LayerTest::CalculateRefs();
    if (!refs.empty()) {
        auto out = refs[0].second;
        expected_output.reserve(out.size());
        for (auto val : out) {
            expected_output.push_back(val);
        }
    }
    return refs;
}

std::vector<InferenceEngine::Blob::Ptr> ConvertColorI420AccuracyTest::GetOutputs() {
    auto outputs = ConvertColorI420LayerTest::GetOutputs();
    if (!outputs.empty()) {
        actual_output = InferenceEngine::Blob::Ptr(outputs[0]);
    }
    return outputs;
}


} // namespace LayerTestsDefinitions
