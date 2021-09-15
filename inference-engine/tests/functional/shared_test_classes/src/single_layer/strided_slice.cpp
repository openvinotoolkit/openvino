// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

#include "shared_test_classes/single_layer/strided_slice.hpp"

namespace LayerTestsDefinitions {

std::string StridedSliceLayerTest::getTestCaseName(const testing::TestParamInfo<StridedSliceParams> &obj) {
    StridedSliceSpecificParams params;
    InferenceEngine::Precision netPrc;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(params, netPrc, inPrc, outPrc, inLayout, outLayout, targetName, additionalConfig) = obj.param;
    std::ostringstream result;
    result << "inShape=" << CommonTestUtils::vec2str(params.inputShape) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "begin=" << CommonTestUtils::vec2str(params.begin) << "_";
    result << "end=" << CommonTestUtils::vec2str(params.end) << "_";
    result << "stride=" << CommonTestUtils::vec2str(params.strides) << "_";
    result << "begin_m=" << CommonTestUtils::vec2str(params.beginMask) << "_";
    result << "end_m=" << CommonTestUtils::vec2str(params.endMask) << "_";
    result << "new_axis_m=" << (params.newAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.newAxisMask)) << "_";
    result << "shrink_m=" << (params.shrinkAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.shrinkAxisMask)) << "_";
    result << "ellipsis_m=" << (params.ellipsisAxisMask.empty() ? "def" : CommonTestUtils::vec2str(params.ellipsisAxisMask)) << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void StridedSliceLayerTest::SetUp() {
    StridedSliceSpecificParams ssParams;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additionalConfig;
    std::tie(ssParams, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, additionalConfig) = this->GetParam();
    configuration.insert(additionalConfig.begin(), additionalConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {ssParams.inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto ss = ngraph::builder::makeStridedSlice(paramOuts[0], ssParams.begin, ssParams.end, ssParams.strides, ngPrc, ssParams.beginMask,
                                                ssParams.endMask, ssParams.newAxisMask, ssParams.shrinkAxisMask, ssParams.ellipsisAxisMask);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ss)};
    function = std::make_shared<ngraph::Function>(results, params, "StridedSlice");
}

std::string Slice8LayerTest::getTestCaseName(const testing::TestParamInfo<Slice8Params> &obj) {
    std::vector<ov::test::InputShape> shapes;
    Slice8SpecificParams params;
    ov::element::Type_t netPrecision;
    std::string targetName;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapes, params, netPrecision, targetName, additionalConfig) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << CommonTestUtils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
    }
    result << "begin="   << CommonTestUtils::vec2str(params.begin) << "_";
    result << "end="     << CommonTestUtils::vec2str(params.end) << "_";
    result << "stride="  << CommonTestUtils::vec2str(params.strides) << "_";
    result << "axes="    << CommonTestUtils::vec2str(params.axes) << "_";
    result << "netPRC="  << netPrecision << "_";
    result << "trgDev="  << targetName;
    return result.str();
}

void Slice8LayerTest::SetUp() {
    std::vector<ov::test::InputShape> shapes;
    Slice8SpecificParams sliceParams;
    ov::test::ElementType netPrecision;
    std::map<std::string, std::string> additionalConfig;
    std::tie(shapes, sliceParams, netPrecision, targetDevice, additionalConfig) = this->GetParam();

    configuration.insert(additionalConfig.begin(), additionalConfig.end());
    init_input_shapes(shapes);
    auto params = ngraph::builder::makeDynamicParams(netPrecision, inputDynamicShapes);
    auto sliceOp = ngraph::builder::makeSlice(params[0], sliceParams.begin, sliceParams.end, sliceParams.strides, sliceParams.axes, netPrecision);

    ov::ResultVector results;
    for (int i = 0; i < sliceOp->get_output_size(); i++)
         results.push_back(std::make_shared<ov::op::v0::Result>(sliceOp->output(i)));
    function = std::make_shared<ngraph::Function>(results, params, "Slice-8");
}

}  // namespace LayerTestsDefinitions
