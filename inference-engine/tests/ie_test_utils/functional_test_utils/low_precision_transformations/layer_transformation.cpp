// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#include "ie_util_internal.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"


namespace LayerTestsUtils {
    InferenceEngine::details::LowPrecisionTransformations LayerTransformation::getLowPrecisionTransformations(
        const InferenceEngine::details::LayerTransformation::Params& params) const {
        if (targetDevice == "CPU") {
            return InferenceEngine::details::LowPrecisionTransformer::getAllTransformations(params).
                add<InferenceEngine::details::ConvolutionTransformation>(InferenceEngine::details::LayerTransformation::Params(params).
                    setPrecisionsOnActivations({ InferenceEngine::Precision::U8 }), "Convolution").
                addCleanup<InferenceEngine::details::ScaleShiftToConvolutionTransformation>(
                    InferenceEngine::details::LayerTransformation::Params(params).setPrecisionsOnActivations({ InferenceEngine::Precision::U8 }),
                    "ScaleShift");
        } else if (targetDevice == "GPU") {
            return InferenceEngine::details::LowPrecisionTransformer::getAllTransformations(params);
        } else {
            THROW_IE_EXCEPTION << "unknown target device " << targetDevice;
        }
    }

    InferenceEngine::details::LowPrecisionTransformer LayerTransformation::getLowPrecisionTransformer(
        const InferenceEngine::details::LayerTransformation::Params& params) const {
        InferenceEngine::details::LowPrecisionTransformer transformer(getLowPrecisionTransformations(params));
        return transformer;
    }

    InferenceEngine::CNNNetwork LayerTransformation::transform() {
        InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = cloneNet(InferenceEngine::CNNNetwork(function));

        InferenceEngine::details::LayerTransformation::Params params = InferenceEngine::details::LayerTransformation::Params(
            true,  // updatePrecisions
            true,  // quantizeOutputs
            true,  // weightsToConst
            InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::UpdateLevel,  // quantizedTensorAlignmentOnActivations
            InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,  // quantizedTensorAlignmentOnWeights
            true,  // roundQuantizedValues
            true,  // updateBiases
            true); // supportAsymmetricQuantization
        auto transformer = getLowPrecisionTransformer(params);
        transformer.transform(*cnnNetworkImp);

        return InferenceEngine::CNNNetwork(cnnNetworkImp);
    }
}  // namespace LayerTestsUtils
