// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/layer_test_utils.hpp"
#include "low_precision_transformations/transformer.hpp"

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
    static InferenceEngine::details::LayerTransformation::Params createParamU8I8();
    static InferenceEngine::details::LayerTransformation::Params createParamU8U8();
    static InferenceEngine::details::LayerTransformation::Params createParamI8I8();
    static InferenceEngine::details::LayerTransformation::Params createParamCpu();
    static InferenceEngine::details::LayerTransformation::Params createParamGpu();
};

template <typename T>
class LayerTransformation : public testing::WithParamInterface<T>, public LayerTestsUtils::LayerTestsCommon {
public:
    InferenceEngine::details::LowPrecisionTransformations getLowPrecisionTransformations(
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

    InferenceEngine::details::LowPrecisionTransformer getLowPrecisionTransformer(
        const InferenceEngine::details::LayerTransformation::Params& params) const {
        InferenceEngine::details::LowPrecisionTransformer transformer(getLowPrecisionTransformations(params));
        return transformer;
    }

    InferenceEngine::CNNNetwork transform() {
        return transform(LayerTransformationParamsFactory::createParamCpu());
    }

    InferenceEngine::CNNNetwork transform(InferenceEngine::details::LayerTransformation::Params& params) {
        InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = cloneNet(InferenceEngine::CNNNetwork(function));

        auto transformer = getLowPrecisionTransformer(params);
        transformer.transform(*cnnNetworkImp);

        return InferenceEngine::CNNNetwork(cnnNetworkImp);
    }

    InferenceEngine::CNNNetwork transform(const InferenceEngine::details::LowPrecisionTransformations& transformations) {
        InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = cloneNet(InferenceEngine::CNNNetwork(function));

        InferenceEngine::details::LowPrecisionTransformer transformer(transformations);
        transformer.transform(*cnnNetworkImp);

        return InferenceEngine::CNNNetwork(cnnNetworkImp);
    }

    static std::string toString(const InferenceEngine::details::LayerTransformation::Params& params) {
        std::ostringstream result;
        result <<
            (params.supportAsymmetricQuantization ? "asymmetric" : "symmetric") << "_" <<
            params.precisionsOnActivations << "_" <<
            params.precisionsOnWeights << "_" <<
            params.quantizedTensorAlignmentOnActivations;

        return result.str();
    }
};

}  // namespace LayerTestsUtils
