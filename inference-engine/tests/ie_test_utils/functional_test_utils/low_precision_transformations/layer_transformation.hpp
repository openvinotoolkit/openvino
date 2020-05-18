// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "functional_test_utils/layer_test_utils.hpp"
#include "low_precision_transformations/transformer.hpp"

namespace LayerTestsUtils {

class LayerTransformation : public testing::WithParamInterface<LayerTestsUtils::basicParams>, public LayerTestsUtils::LayerTestsCommon {
public:
    InferenceEngine::details::LowPrecisionTransformations getLowPrecisionTransformations(
        const InferenceEngine::details::LayerTransformation::Params& params) const;

    InferenceEngine::details::LowPrecisionTransformer getLowPrecisionTransformer(
        const InferenceEngine::details::LayerTransformation::Params& params) const;

    InferenceEngine::CNNNetwork transform();
};

}  // namespace LayerTestsUtils
