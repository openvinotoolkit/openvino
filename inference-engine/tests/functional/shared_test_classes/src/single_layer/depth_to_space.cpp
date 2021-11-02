// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/depth_to_space.hpp"

namespace LayerTestsDefinitions {

using namespace ngraph::opset3;

static inline std::string DepthToSpaceModeToString(const DepthToSpace::DepthToSpaceMode& mode) {
    static std::map<DepthToSpace::DepthToSpaceMode, std::string> names = {
        {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    auto i = names.find(mode);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported DepthToSpaceMode");
}

std::string DepthToSpaceLayerTest::getTestCaseName(const testing::TestParamInfo<depthToSpaceParamsTuple> &obj) {
    depthToSpaceSpecificParamsTuple d2sParams;
    std::vector<size_t> inShape;
    DepthToSpace::DepthToSpaceMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::string targetName;
    std::tie(d2sParams, inShape, inputPrecision, targetName) = obj.param;
    std::tie(mode, blockSize) = d2sParams;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inShape) << "_";
    result << "inPrc=" << inputPrecision.name() << "_";
    result << "M=" << DepthToSpaceModeToString(mode) << "_";
    result << "BS=" << blockSize << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void DepthToSpaceLayerTest::SetUp() {
    depthToSpaceSpecificParamsTuple d2sParams;
    std::vector<size_t> inShape;
    DepthToSpace::DepthToSpaceMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::tie(d2sParams, inShape, inputPrecision, targetDevice) = this->GetParam();
    std::tie(mode, blockSize) = d2sParams;
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto params = ngraph::builder::makeParams(inPrc, {inShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto d2s = ngraph::builder::makeDepthToSpace(paramOuts[0], mode, blockSize);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(d2s)};
    function = std::make_shared<ngraph::Function>(results, params, "DepthToSpace");
}
}  // namespace LayerTestsDefinitions
