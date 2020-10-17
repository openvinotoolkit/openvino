// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include <legacy/ie_util_internal.hpp>
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "low_precision_transformations/transformer.hpp"
#include <transformations/low_precision/transformer.hpp>

namespace LayerTestsUtils {

class LayerTransformationParamsFactory {
public:
    static InferenceEngine::details::LayerTransformation::Params createParamsU8I8();
    static InferenceEngine::details::LayerTransformation::Params createParamsU8U8();
    static InferenceEngine::details::LayerTransformation::Params createParamsI8I8();
    static InferenceEngine::details::LayerTransformation::Params createParams();
};

class LayerTransformationParamsNGraphFactory {
public:
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsU8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParamsI8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params createParams();
};

IE_SUPPRESS_DEPRECATED_START

class LayerTransformation : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static ngraph::pass::low_precision::LayerTransformation::Params toNGraph(const InferenceEngine::details::LayerTransformation::Params& params);
    static InferenceEngine::details::LayerTransformation::Params toCNNNetwork(const ngraph::pass::low_precision::LayerTransformation::Params& params);
    static InferenceEngine::Precision toCNNNetwork(const ngraph::element::Type_t precision);

protected:
    LayerTransformation();

    static InferenceEngine::Blob::Ptr GenerateInput(
        const ngraph::element::Type_t precision,
        const InferenceEngine::TensorDesc& tensorDesc,
        const float k = 1.f);

    InferenceEngine::details::LowPrecisionTransformations getLowPrecisionTransformations(
        const InferenceEngine::details::LayerTransformation::Params& params) const;

    ngraph::pass::low_precision::LowPrecisionTransformations getLowPrecisionTransformationsNGraph(
        const ngraph::pass::low_precision::LayerTransformation::Params& params) const;

    InferenceEngine::details::LowPrecisionTransformer getLowPrecisionTransformer(
        const InferenceEngine::details::LayerTransformation::Params& params) const;

    ngraph::pass::low_precision::LowPrecisionTransformer getLowPrecisionTransformerNGraph(
        const ngraph::pass::low_precision::LayerTransformation::Params& params) const;

    std::shared_ptr<ngraph::Function> transformNGraph(
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::pass::low_precision::LowPrecisionTransformations additionalTransformations = {});

    InferenceEngine::CNNNetwork transform(const InferenceEngine::details::LowPrecisionTransformations& transformations);

    static void checkPrecisions(const InferenceEngine::CNNLayer& layer, const InferenceEngine::Precision& expectedPrecision);

    static void checkPrecisions(
        const InferenceEngine::CNNLayer& layer,
        const std::vector<std::vector<InferenceEngine::Precision>>& expectedInputPrecisions,
        const std::vector<InferenceEngine::Precision>& expectedOutputPrecisions,
        const bool asymmetricQuantizationOnData = false,
        const bool asymmetricQuantizationOnWeights = false);

    static std::pair<float, float> getQuantizationInterval(const ngraph::element::Type_t precision);

    static std::string toString(const InferenceEngine::details::LayerTransformation::Params& params);

    static std::string toString(const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static InferenceEngine::Precision getDeviceInternalPrecision(const InferenceEngine::Precision precision);

    static std::string getTestCaseNameByParams(
        const InferenceEngine::Precision precision,
        const InferenceEngine::SizeVector& inputShapes,
        const std::string& targetDevice,
        const InferenceEngine::details::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShapes,
        const std::string& targetDevice,
        const ngraph::pass::low_precision::LayerTransformation::Params& params);

    static bool fakeQuantizeExists(const InferenceEngine::ICNNNetwork& network);
};

IE_SUPPRESS_DEPRECATED_END

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    // TODO: refactor: CNNNetwork LPT is detected
    InferenceEngine::details::LayerTransformation::Params> LayerTransformationParams;

}  // namespace LayerTestsUtils
