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

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "ie_util_internal.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

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

InferenceEngine::Blob::Ptr LayerTransformation::GenerateInput(
    const InferenceEngine::Precision precision,
    const InferenceEngine::TensorDesc& tensorDesc,
    const float k) {
    const auto interval = getQuantizationInterval(precision);
    const float low = interval.first / k;
    const float hight = interval.second / k;

    return FuncTestUtils::createAndFillBlobConsistently(tensorDesc, hight - low, static_cast<int32_t>(low), 1ul);
}

LayerTransformation::LayerTransformation() {
    threshold = 0.05;
}

InferenceEngine::details::LowPrecisionTransformer LayerTransformation::getLowPrecisionTransformer(
    const InferenceEngine::details::LayerTransformation::Params& params) const {
    InferenceEngine::details::LowPrecisionTransformer transformer(getLowPrecisionTransformations(params));
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

}  // namespace LayerTestsUtils
