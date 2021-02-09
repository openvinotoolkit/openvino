// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/deformable_convolution.hpp"

namespace LayerTestsDefinitions {

std::string DeformableConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<deformableConvLayerTestParamsSet> obj) {
    deformableConvSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) =
        obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector deformable_vals, kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t groups, deformable_groups, convOutChannels;
    std::tie(deformable_vals, kernel, stride, padBegin, padEnd, dilation, groups, deformable_groups, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "DV" << CommonTestUtils::vec2str(deformable_vals) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "G=" << groups << "_";
    result << "DG=" << deformable_groups << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
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
        const std::string name = info.name();
        if (name == "a_data")
        {

        }
        else if (name == "a_data")
        {
            
        }
        else if (name == "b_defor_vals")
        {
            blobPtr = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 0, 0, 0, 7235346); // TODO
        }
        else if (name == "c_kernel_vals")
        {
            
        }
        else if (name == "d_stride_vals")
        {
            
        }
        else if (name == "e_pads_begin_vals")
        {
            
        }
        else if (name == "f_pads_begin_vals")
        {
            
        }
        else if (name == "g_dilation_vals")
        {
            
        }
        else if (name == "h_pad_type_vals")
        {
            
        }
        else if (name == "i_groups_vals")
        {
            
        }
        else if (name == "j_deformable_groups_vals")
        {
            
        }
        return blobPtr;
}

void DeformableConvolutionLayerTest::SetUp() {
    deformableConvSpecificParams convParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) =
        this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector deformable_vals, kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t groups, deformable_groups, convOutChannels;
    std::tie(deformable_vals, kernel, stride, padBegin, padEnd, dilation, groups, deformable_groups, convOutChannels, padType) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, deformable_vals, kernel});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    
    auto data = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(inputShape));
    data->set_friendly_name("a_data");
    auto defor_vals = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(deformable_vals));
    defor_vals->set_friendly_name("b_defor_vals");
    auto kernel_vals = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(kernel));
    kernel_vals->set_friendly_name("c_kernel_vals");
    auto stride_vals = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(stride));
    stride_vals->set_friendly_name("d_stride_vals");
    auto pads_begin_vals = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(padBegin));
    pad_begins_vals->set_friendly_name("e_pads_begin_vals");
    auto pads_end_vals = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(padEnd));
    pad_end_vals->set_friendly_name("f_pads_begin_vals");
    auto dilation_vals = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(dilation));
    dilation_vals->set_friendly_name("g_dilation_vals");
    auto pad_type_vals = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(padType));
    pad_type_vals->set_friendly_name("h_pad_type_vals");
    auto groups_vals = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(groups));
    groups_vals->set_friendly_name("i_groups_vals");
    auto deformable_groups = std::make_shared<ngraph::op::Parameter>(netPrecision, ngraph::Shape(deformable_groups));
    deformable_groups->set_friendly_name("j_deformable_groups_vals");

    auto deformable_conv = std::make_shared<ngraph::opset1::DeformableConvolution>(paramOuts[0], paramOuts[1], paramOuts[2],
                                                              stride, padBegin, padEnd, dilation, padType, groups, deformable_groups);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(deformable_conv)};
    function = std::make_shared<ngraph::Function>(results, params, "deformable_convolution");
}
}  // namespace LayerTestsDefinitions
