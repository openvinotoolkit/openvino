// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformer_single_layer_tests.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/fully_connected.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"

LowPrecisionTransformations SingleLayerTestModel::getLowPrecisionTransformations(const LayerTransformation::Params& params) const {
    if (device_name == "CPU") {
        return LowPrecisionTransformer::getAllTransformations(params).
            add<ConvolutionTransformation>(LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }), "Convolution").
            addCleanup<ScaleShiftToConvolutionTransformation>(
                LayerTransformation::Params(params).setPrecisionsOnActivations({ Precision::U8 }),
                "ScaleShift");
    } else if (device_name == "GPU") {
        return LowPrecisionTransformer::getAllTransformations(params);
    } else {
        THROW_IE_EXCEPTION << "unknown plugin " << device_name;
    }
}

LowPrecisionTransformer SingleLayerTestModel::getLowPrecisionTransformer(const LayerTransformation::Params& params) const {
    LowPrecisionTransformer transformer(getLowPrecisionTransformations(params));
    return transformer;
}
