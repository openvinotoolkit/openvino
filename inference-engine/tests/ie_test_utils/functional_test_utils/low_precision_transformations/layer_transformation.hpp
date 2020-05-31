// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/layer_test_utils.hpp"
#include "low_precision_transformations/transformer.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include "ie_util_internal.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

namespace LayerTestsUtils {

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    InferenceEngine::details::LayerTransformation::Params> LayerTransformationParams;

class LayerTransformationParamsFactory {
public:
    static InferenceEngine::details::LayerTransformation::Params createParamsU8I8();
    static InferenceEngine::details::LayerTransformation::Params createParamsU8U8();
    static InferenceEngine::details::LayerTransformation::Params createParamsI8I8();
    static InferenceEngine::details::LayerTransformation::Params createParams();
};

IE_SUPPRESS_DEPRECATED_START

class LayerTransformation : public LayerTestsUtils::LayerTestsCommon {
protected:
    static InferenceEngine::Blob::Ptr GenerateInput(
        const InferenceEngine::Precision precision,
        const InferenceEngine::TensorDesc& tensorDesc,
        const float k = 1.f);

    InferenceEngine::details::LowPrecisionTransformations getLowPrecisionTransformations(
        const InferenceEngine::details::LayerTransformation::Params& params) const;

    InferenceEngine::details::LowPrecisionTransformer getLowPrecisionTransformer(
        const InferenceEngine::details::LayerTransformation::Params& params) const;

    InferenceEngine::CNNNetwork transform(InferenceEngine::details::LayerTransformation::Params& params);

    InferenceEngine::CNNNetwork transform(const InferenceEngine::details::LowPrecisionTransformations& transformations);

    static void checkPrecisions(const InferenceEngine::CNNLayer& layer, const InferenceEngine::Precision& expectedPrecision);

    static void checkPrecisions(
        const InferenceEngine::CNNLayer& layer,
        const std::vector<std::vector<InferenceEngine::Precision>>& expectedInputPrecisions,
        const std::vector<InferenceEngine::Precision>& expectedOutputPrecisions);

    static std::pair<float, float> getQuantizationInterval(const InferenceEngine::Precision precision);

    static std::string toString(const InferenceEngine::details::LayerTransformation::Params& params);

    static InferenceEngine::Precision getDeviceInternalPrecision(const InferenceEngine::Precision precision);
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace LayerTestsUtils
