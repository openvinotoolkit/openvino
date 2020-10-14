// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/depth_to_space_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/builders.hpp"

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/depth_to_space_fusion.hpp>
#include <ngraph/op/depth_to_space.hpp>

#include "ngraph_functions/low_precision_transformations/depth_to_space_function.hpp"

using namespace ngraph::opset1;

namespace LayerTestsDefinitions {

std::string DepthToSpaceTransformation::getTestCaseName(testing::TestParamInfo<DepthToSpaceTransformationParams> obj) {
    static std::map<DepthToSpace::DepthToSpaceMode, std::string> names = {
        {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(precision, inputShape, targetDevice, mode, blockSize, version) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(precision, inputShape, targetDevice, params, version) <<
        "_" << names[mode] << "_" << blockSize;
    return result.str();
}

void DepthToSpaceTransformation::SetUp() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(precision, inputShape, targetDevice, mode, blockSize, version) = this->GetParam();

    if (inputShape.size() != 4ul) {
        THROW_IE_EXCEPTION << "not supported input shape size " << inputShape.size();
    }

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::DepthToSpaceFunction::getOriginal(precision, inputShape, mode, blockSize);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void DepthToSpaceTransformation::validate() {
    ngraph::element::Type precision;
    ngraph::Shape inputShape;
    std::string targetDevice;
    DepthToSpace::DepthToSpaceMode mode;
    size_t blockSize;
    auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(precision, inputShape, targetDevice, mode, blockSize, version) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(toCNNNetwork(params));

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    EXPECT_EQ(1ul, outputLayer->insData.size());
    const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr);
    const InferenceEngine::CNNLayerPtr depthToSpace = getCreatorLayer(insData).lock();
    EXPECT_TRUE(depthToSpace != nullptr);
    EXPECT_EQ("DepthToSpace", depthToSpace->type);

    if (params.updatePrecisions) {
        const InferenceEngine::Precision precision = depthToSpace->outData[0]->getTensorDesc().getPrecision();
        EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(DepthToSpaceTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
