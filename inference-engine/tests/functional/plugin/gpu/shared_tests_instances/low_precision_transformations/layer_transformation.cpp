// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>
#include <net_pass.h>
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ie_util_internal.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

namespace LayerTestsUtils {

InferenceEngine::details::LowPrecisionTransformations LayerTransformation::getLowPrecisionTransformations(
    const InferenceEngine::details::LayerTransformation::Params& params) const {
    return InferenceEngine::details::LowPrecisionTransformer::getAllTransformations(params);
}

InferenceEngine::Precision LayerTransformation::getDeviceInternalPrecision(const InferenceEngine::Precision precision) {
    if (precision == InferenceEngine::Precision::FP16) {
        return InferenceEngine::Precision::FP32;
    }

    return precision;
}

InferenceEngine::CNNNetwork LayerTransformation::transform(InferenceEngine::details::LayerTransformation::Params& params) {
    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = cloneNet(InferenceEngine::CNNNetwork(function));

    InferenceEngine::NetPass::ConvertPrecision(*cnnNetworkImp, InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32);

    auto transformer = getLowPrecisionTransformer(params);
    transformer.transform(*cnnNetworkImp);

    return InferenceEngine::CNNNetwork(cnnNetworkImp);
}

InferenceEngine::CNNNetwork LayerTransformation::transform(const InferenceEngine::details::LowPrecisionTransformations& transformations) {
    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = cloneNet(InferenceEngine::CNNNetwork(function));

    InferenceEngine::NetPass::ConvertPrecision(*cnnNetworkImp, InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32);

    InferenceEngine::details::LowPrecisionTransformer transformer(transformations);
    transformer.transform(*cnnNetworkImp);

    return InferenceEngine::CNNNetwork(cnnNetworkImp);
}

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParams() {
    return InferenceEngine::details::LayerTransformation::Params(
        true,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        true,
        true,
        true);
}
}  // namespace LayerTestsUtils
