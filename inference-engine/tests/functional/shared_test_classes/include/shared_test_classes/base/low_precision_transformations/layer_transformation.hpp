// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include <low_precision/transformer.hpp>

namespace LayerTestsUtils {

class LayerTransformationParamsNGraphFactory {
public:
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8AndI8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsI8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParams();
};

class LayerTransformationParamsFactory : public LayerTransformationParamsNGraphFactory {
};

class LayerTransformation : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    LayerTransformation();

    static InferenceEngine::Blob::Ptr GenerateInput(
        const ngraph::element::Type precision,
        const InferenceEngine::TensorDesc& tensorDesc,
        const float k = 1.f);

    ngraph::pass::low_precision::LowPrecisionTransformations getLowPrecisionTransformationsNGraph(
        const ngraph::pass::low_precision::LayerTransformation::Params& params) const;

    ngraph::pass::low_precision::LowPrecisionTransformer getLowPrecisionTransformerNGraph(
        const ngraph::pass::low_precision::LayerTransformation::Params& params) const;

    std::shared_ptr<ngraph::Function> transformNGraph(
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::pass::low_precision::LowPrecisionTransformations& transformations);

    static std::pair<float, float> getQuantizationInterval(const ngraph::element::Type precision);

    static std::string toString(const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static InferenceEngine::Precision getDeviceInternalPrecision(const InferenceEngine::Precision precision);

    static std::string getTestCaseNameByParams(
        const InferenceEngine::Precision precision,
        const InferenceEngine::SizeVector& inputShapes,
        const std::string& targetDevice,
        const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShapes,
        const std::string& targetDevice,
        const ngraph::pass::low_precision::LayerTransformation::Params& params);
};

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

}  // namespace LayerTestsUtils
