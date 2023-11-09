// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
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
    std::vector<size_t> inShape;
    DepthToSpace::DepthToSpaceMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::string targetName;
    std::tie(inShape, inputPrecision, mode, blockSize, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inShape) << "_";
    result << "inPrc=" << inputPrecision.name() << "_";
    result << "M=" << DepthToSpaceModeToString(mode) << "_";
    result << "BS=" << blockSize << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void DepthToSpaceLayerTest::SetUp() {
    std::vector<size_t> inShape;
    DepthToSpace::DepthToSpaceMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::tie(inShape, inputPrecision, mode, blockSize, targetDevice) = this->GetParam();
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(inPrc, ov::Shape(inShape))};
    auto d2s = std::make_shared<ov::op::v0::DepthToSpace>(params[0], mode, blockSize);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(d2s)};
    function = std::make_shared<ngraph::Function>(results, params, "DepthToSpace");
}
}  // namespace LayerTestsDefinitions
