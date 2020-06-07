// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <sstream>
#include <tuple>

#include "ie_util_internal.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "low_precision_transformations/transformer.hpp"

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

class LayerTransformation : public LayerTestsUtils::LayerTestsCommon {
protected:
    InferenceEngine::details::LowPrecisionTransformations getLowPrecisionTransformations(
        const InferenceEngine::details::LayerTransformation::Params& params) const;

    InferenceEngine::details::LowPrecisionTransformer getLowPrecisionTransformer(
        const InferenceEngine::details::LayerTransformation::Params& params) const;

    InferenceEngine::CNNNetwork transform(InferenceEngine::details::LayerTransformation::Params& params);

    InferenceEngine::CNNNetwork transform(const InferenceEngine::details::LowPrecisionTransformations& transformations);

    static void checkParentPrecision(const InferenceEngine::CNNLayerPtr& layer, const bool lowPrecision);

    static std::string toString(const InferenceEngine::details::LayerTransformation::Params& params);
};

}  // namespace LayerTestsUtils
