// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <unordered_set>

#include <ie_core.hpp>
#include <net_pass.h>
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "ie_util_internal.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

using namespace InferenceEngine;

namespace LayerTestsUtils {

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsU8I8() {
    return InferenceEngine::details::LayerTransformation::Params(
        true,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::U8 },
        { InferenceEngine::Precision::I8 });
}

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsU8U8() {
    return InferenceEngine::details::LayerTransformation::Params(
        true,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::U8 },
        { InferenceEngine::Precision::U8 });
}

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsI8I8() {
    return InferenceEngine::details::LayerTransformation::Params(
        true,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::I8 },
        { InferenceEngine::Precision::I8 });
}

LayerTransformation::LayerTransformation() {
    threshold = 0.05;
}

void LayerTransformation::ConfigurePlugin(const LptVersion lptVersion) {
    switch (lptVersion) {
        case LptVersion::cnnNetwork: {
            configuration[PluginConfigInternalParams::KEY_LP_TRANSFORMS_VERSION] = PluginConfigInternalParams::LP_TRANSFORMS_CNNNETWORK;
            break;
        }
        case LptVersion::nGraph: {
            configuration[PluginConfigInternalParams::KEY_LP_TRANSFORMS_VERSION] = PluginConfigInternalParams::LP_TRANSFORMS_NGRAPH;
            break;
        }
        default: {
            THROW_IE_EXCEPTION << "unexpected LPT version " << lptVersion;
        }
    }
}

InferenceEngine::Blob::Ptr LayerTransformation::GenerateInput(
    const InferenceEngine::Precision precision,
    const InferenceEngine::TensorDesc& tensorDesc,
    const float k) {
    const auto interval = getQuantizationInterval(precision);
    const float low = interval.first / k;
    const float hight = interval.second / k;

    return FuncTestUtils::createAndFillBlobConsistently(tensorDesc, hight - low, static_cast<int32_t>(low), 1ul);
}

InferenceEngine::details::LowPrecisionTransformer LayerTransformation::getLowPrecisionTransformer(
    const InferenceEngine::details::LayerTransformation::Params& params) const {
    InferenceEngine::details::LowPrecisionTransformer transformer(getLowPrecisionTransformations(params));
    return transformer;
}

ngraph::pass::low_precision::LowPrecisionTransformer LayerTransformation::getLowPrecisionTransformerNGraph(
    const ngraph::pass::low_precision::LayerTransformation::Params& params) const {
    ngraph::pass::low_precision::LowPrecisionTransformer transformer(getLowPrecisionTransformationsNGraph(params));
    return transformer;
}

IE_SUPPRESS_DEPRECATED_START

void LayerTransformation::checkPrecisions(const InferenceEngine::CNNLayer& layer, const InferenceEngine::Precision& expectedPrecision) {
    for (const InferenceEngine::DataWeakPtr insDataWeak : layer.insData) {
        const InferenceEngine::DataPtr insData = insDataWeak.lock();
        EXPECT_TRUE(insData != nullptr) << "insert data is nullable";
        const InferenceEngine::Precision inputPrecision = insData->getTensorDesc().getPrecision();
        EXPECT_EQ(getDeviceInternalPrecision(expectedPrecision), inputPrecision) <<
            "expected input precision " << getDeviceInternalPrecision(expectedPrecision) << " actual precision " << inputPrecision;
    }

    for (const InferenceEngine::DataPtr outData : layer.outData) {
        const InferenceEngine::Precision outputPrecision = outData->getTensorDesc().getPrecision();
        EXPECT_EQ(getDeviceInternalPrecision(expectedPrecision), outputPrecision) <<
            "expected output precision " << getDeviceInternalPrecision(expectedPrecision) << " actual precision " << outputPrecision;
    }
}

void LayerTransformation::checkPrecisions(
    const InferenceEngine::CNNLayer& layer,
    const std::vector<std::vector<InferenceEngine::Precision>>& expectedInputPrecisions,
    const std::vector<InferenceEngine::Precision>& expectedOutputPrecisions,
    const bool asymmetricQuantizationOnData,
    const bool asymmetricQuantizationOnWeights) {
    EXPECT_EQ(expectedInputPrecisions.size(), layer.insData.size()) << "insert data count is no expected: " << layer.insData.size();

    const auto checkPrecision = [](
        const InferenceEngine::CNNLayer& layer,
        const std::vector<InferenceEngine::Precision>& expectedPrecisions,
        const size_t index,
        const bool input) {
        const InferenceEngine::DataPtr data = input ? layer.insData[index].lock() : layer.outData[index];
        EXPECT_TRUE(data != nullptr) << "data is nullable";
        const InferenceEngine::Precision actualPrecision = data->getTensorDesc().getPrecision();

        EXPECT_FALSE(std::all_of(
            expectedPrecisions.begin(),
            expectedPrecisions.end(),
            [&](const InferenceEngine::Precision precision) { return getDeviceInternalPrecision(precision) != actualPrecision; })) <<
            "expected precisions on " << index << (input ? " input" : " output") << " port " << expectedPrecisions <<
            " actual precision " << actualPrecision;
    };

    if (asymmetricQuantizationOnData || asymmetricQuantizationOnWeights) {
        if (asymmetricQuantizationOnData) {
            const InferenceEngine::CNNLayerPtr parentOnData = InferenceEngine::details::CNNNetworkHelper::getParent(layer, 0);
            checkPrecision(*parentOnData, expectedInputPrecisions[0], 0, true);
        } else {
            checkPrecision(layer, expectedInputPrecisions[0], 0, true);
        }

        if (asymmetricQuantizationOnWeights) {
            const InferenceEngine::CNNLayerPtr parentOnWeights = InferenceEngine::details::CNNNetworkHelper::getParent(layer, 1);
            checkPrecision(*parentOnWeights, expectedInputPrecisions[1], 1, true);
        } else {
            checkPrecision(layer, expectedInputPrecisions[1], 1, true);
        }
    } else {
        for (size_t inputIndex = 0ul; inputIndex < layer.insData.size(); ++inputIndex) {
            checkPrecision(layer, expectedInputPrecisions[inputIndex], inputIndex, true);
        }
    }

    checkPrecision(layer, expectedOutputPrecisions, 0, false);
}

IE_SUPPRESS_DEPRECATED_END

std::pair<float, float> LayerTransformation::getQuantizationInterval(const InferenceEngine::Precision precision) {
    const bool unsignedInterval = precision == InferenceEngine::Precision::U8;
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

std::string LayerTransformation::toString(const InferenceEngine::details::LayerTransformation::Params& params) {
    std::ostringstream result;
    result <<
        (params.supportAsymmetricQuantization ? "asymmetric_" : "symmetric_") <<
        (params.updatePrecisions ? "" : "notUpdatePrecisions_") <<
        params.precisionsOnActivations << "_" <<
        params.precisionsOnWeights << "_" <<
        params.quantizedTensorAlignmentOnActivations;

    return result.str();
}

std::string LayerTransformation::getTestCaseNameByParams(
    const InferenceEngine::Precision netPrecision,
    const InferenceEngine::SizeVector& inputShapes,
    const std::string targetDevice,
    const InferenceEngine::details::LayerTransformation::Params& params,
    const LayerTestsUtils::LayerTransformation::LptVersion version) {
    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << version << "_" << toString(params);
    return result.str();
}

IE_SUPPRESS_DEPRECATED_START

bool LayerTransformation::fakeQuantizeExists(const InferenceEngine::ICNNNetwork& network) {
    auto it = InferenceEngine::details::CNNNetworkIterator(&network);
    auto end = details::CNNNetworkIterator();
    while (it != end) {
        if (((*it)->type == "FakeQuantize") && (InferenceEngine::details::QuantizationDetails::isSupportedLevel((*it)->GetParamAsUInt("levels")))) {
            return true;
        }
        it++;
    }

    return false;
}

IE_SUPPRESS_DEPRECATED_END

ngraph::element::Type toNGraph(const InferenceEngine::Precision precision) {
    switch (precision) {
        case InferenceEngine::Precision::U8: {
            return ngraph::element::u8;
        }
        case InferenceEngine::Precision::I8: {
            return ngraph::element::i8;
        }
        default: {
            THROW_IE_EXCEPTION << "unknown precision " << precision;
        }
    }
}

std::vector<ngraph::element::Type> toNGraph(const std::vector<InferenceEngine::Precision>& precisions) {
    std::vector<ngraph::element::Type> resultPrecisions(precisions.size());
    for (size_t i = 0ul; i < precisions.size(); ++i) {
        const InferenceEngine::Precision precision = precisions[i];
        resultPrecisions[i] = toNGraph(precision);
    }
    return resultPrecisions;
}

ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment toNGraph(
    InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment aligment) {
    switch (aligment) {
        case InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::UpdateLevel: {
            return ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment::UpdateLevel;
        }
        case InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None: {
            return ngraph::pass::low_precision::LayerTransformation::QuantizedTensorAlignment::None;
        }
        default: {
            THROW_IE_EXCEPTION << "not supported";
        }
    }
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::toNGraph(const InferenceEngine::details::LayerTransformation::Params& params) {
    const auto precisionsOnActivations = LayerTestsUtils::toNGraph(params.precisionsOnActivations);
    const auto precisionsOnWeights = LayerTestsUtils::toNGraph(params.precisionsOnWeights);
    return ngraph::pass::low_precision::LayerTransformation::Params(
        params.updatePrecisions,
        params.quantizeOutputs,
        params.weightsToConst,
        LayerTestsUtils::toNGraph(params.quantizedTensorAlignmentOnActivations),
        LayerTestsUtils::toNGraph(params.quantizedTensorAlignmentOnWeights),
        params.roundQuantizedValues,
        params.updateBiases,
        params.supportAsymmetricQuantization,
        precisionsOnActivations,
        precisionsOnWeights);
}

}  // namespace LayerTestsUtils
