// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/single_layer/deformable_convolution.hpp"

namespace LayerTestsDefinitions {
std::string DeformableConvolutionLayerTest::getTestCaseName(const testing::TestParamInfo<deformableConvLayerTestParamsSet>& obj) {
    deformableConvSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) =
            obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector offsets, filter, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t groups, deformable_groups, convOutChannels;
    bool with_bilinear_interpolation_pad, with_modulation;
    std::tie(offsets, filter, stride, padBegin, padEnd, dilation, groups, deformable_groups, convOutChannels, padType,
             with_bilinear_interpolation_pad, with_modulation) = convParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "DV" << ov::test::utils::vec2str(offsets) << "_";
    result << "K" << ov::test::utils::vec2str(filter) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "G=" << groups << "_";
    result << "DG=" << deformable_groups << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "BI_PAD=" << with_bilinear_interpolation_pad << "_";
    result << "MODULATION=" << with_modulation << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr DeformableConvolutionLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::Blob::Ptr blobPtr;
    const std::string& name = info.name();
    if (name == "a_data") {
        blobPtr = LayerTestsUtils::LayerTestsCommon::GenerateInput(info);
    } else if (name == "b_offset_vals") {
        blobPtr = FuncTestUtils::createAndFillBlobFloat(info.getTensorDesc(), 2, 0, 10);
    } else if (name == "c_filter_vals") {
        blobPtr = LayerTestsUtils::LayerTestsCommon::GenerateInput(info);
    } else if (name == "c_modulation_scalars") {
        blobPtr = FuncTestUtils::createAndFillBlobFloat(info.getTensorDesc(), 1, 0, 20);
    }
    return blobPtr;
}
void DeformableConvolutionLayerTest::SetUp() {
    deformableConvSpecificParams convParams;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) =
            this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector offsets, filter, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t groups, deformable_groups, convOutChannels;
    bool with_bilinear_interpolation_pad, with_modulation;
    std::tie(offsets, filter, stride, padBegin, padEnd, dilation, groups, deformable_groups, convOutChannels, padType,
             with_bilinear_interpolation_pad, with_modulation) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params;
    for (auto&& shape : {inputShape, offsets, filter}) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape)));
    }
    auto data = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(inputShape));
    data->set_friendly_name("a_data");
    auto offset_vals = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(offsets));
    offset_vals->set_friendly_name("b_offset_vals");
    auto filter_vals = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(filter));
    filter_vals->set_friendly_name("c_filter_vals");
    ngraph::ParameterVector parameters{data, offset_vals, filter_vals};
    std::shared_ptr<ngraph::Node> deformable_conv;
    if (with_modulation) {
        auto modulation_shape = ngraph::Shape(offsets);
        modulation_shape[1] = offsets[1] / 2;
        auto modulation_scalars = std::make_shared<ngraph::op::Parameter>(ngPrc, modulation_shape);
        modulation_scalars->set_friendly_name("c_modulation_scalars");

        deformable_conv = std::make_shared<ngraph::op::v8::DeformableConvolution>(data, offset_vals, filter_vals, modulation_scalars, stride, padBegin,
                                                                                  padEnd, dilation, padType, groups, deformable_groups,
                                                                                  with_bilinear_interpolation_pad);
        parameters.push_back(modulation_scalars);
    } else {
        deformable_conv = std::make_shared<ngraph::op::v8::DeformableConvolution>(data, offset_vals, filter_vals, stride, padBegin, padEnd, dilation,
                                                                                  padType, groups, deformable_groups, with_bilinear_interpolation_pad);
    }

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(deformable_conv)};
    function = std::make_shared<ngraph::Function>(results, parameters, "deformable_convolution");
}
}  // namespace LayerTestsDefinitions
