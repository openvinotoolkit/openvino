// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <queue>
#include <ie_core.hpp>

#include "ngraph/op/op.hpp"
#include <transformations/init_node_info.hpp>
#include "low_precision_transformations/unsqueeze_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/unsqueeze_function.hpp"

namespace LayerTestsDefinitions {

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

InferenceEngine::Blob::Ptr UnsqueezeTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    UnsqueezeTransformationParam squeezeParam;
    std::string targetDevice;

    std::tie(netPrecision, targetDevice, params, version, squeezeParam) = this->GetParam();

    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData = squeezeParam.fakeQuantize;

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

std::string UnsqueezeTransformation::getTestCaseName(testing::TestParamInfo<UnsqueezeTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::string targetDevice;
    UnsqueezeTransformationParam unsqueezeParam;
    std::tie(netPrecision, targetDevice, params, version, unsqueezeParam) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, unsqueezeParam.shape, targetDevice, params, version) << "_" <<
        unsqueezeParam.fakeQuantize << "_" <<
        unsqueezeParam.unsqueezeAxes << "_" <<
        params.updatePrecisions << "_" <<
        unsqueezeParam.shape;

    return result.str();
}
void UnsqueezeTransformation::SetUp() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    UnsqueezeTransformationParam unsqueezeParam;

    std::tie(netPrecision, targetDevice, params, version, unsqueezeParam) = this->GetParam();
    ngraph::element::Type ngraphPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::UnsqueezeFunction::getOriginal(
        ngraphPrecision,
        unsqueezeParam.shape,
        unsqueezeParam.fakeQuantize,
        unsqueezeParam.unsqueezeAxes);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void UnsqueezeTransformation::validate() {
    ngraph::Shape inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    UnsqueezeTransformationParam unsqueezeParam;

    std::tie(netPrecision, targetDevice, params, version, unsqueezeParam) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());
    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
    if (params.updatePrecisions) {
        checkPrecisions(
            *layer,
            { { InferenceEngine::Precision::U8 }, { InferenceEngine::Precision::I8 } },
            { getDeviceInternalPrecision(netPrecision) });
    } else {
        checkPrecisions(*layer, netPrecision);
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(UnsqueezeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
