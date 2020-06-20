// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "ie_util_internal.hpp"
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

IE_SUPPRESS_DEPRECATED_START

class LayerTransformation : public LayerTestsUtils::LayerTestsCommon {
public:
    enum LptVersion {
        cnnNetwork,
        nGraph
    };

protected:
    LayerTransformation();

    void ConfigurePlugin(const LptVersion lptVersion);

    static InferenceEngine::Blob::Ptr GenerateInput(
        const InferenceEngine::Precision precision,
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

    InferenceEngine::CNNNetwork transform(InferenceEngine::details::LayerTransformation::Params& params);

    std::shared_ptr<ngraph::Function> transformNGraph(InferenceEngine::details::LayerTransformation::Params& params);

    InferenceEngine::CNNNetwork transform(const InferenceEngine::details::LowPrecisionTransformations& transformations);

    static void checkPrecisions(const InferenceEngine::CNNLayer& layer, const InferenceEngine::Precision& expectedPrecision);

    static void checkPrecisions(
        const InferenceEngine::CNNLayer& layer,
        const std::vector<std::vector<InferenceEngine::Precision>>& expectedInputPrecisions,
        const std::vector<InferenceEngine::Precision>& expectedOutputPrecisions,
        const bool asymmetricQuantizationOnData = false,
        const bool asymmetricQuantizationOnWeights = false);

    static std::pair<float, float> getQuantizationInterval(const InferenceEngine::Precision precision);

    static std::string toString(const InferenceEngine::details::LayerTransformation::Params& params);

    static InferenceEngine::Precision getDeviceInternalPrecision(const InferenceEngine::Precision precision);

    static ngraph::pass::low_precision::LayerTransformation::Params toNGraph(const InferenceEngine::details::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const InferenceEngine::Precision netPrecision,
        const InferenceEngine::SizeVector& inputShapes,
        const std::string targetDevice,
        const InferenceEngine::details::LayerTransformation::Params& params,
        const LayerTestsUtils::LayerTransformation::LptVersion version);

    static bool fakeQuantizeExists(const InferenceEngine::ICNNNetwork& network);
};

IE_SUPPRESS_DEPRECATED_END

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    // TODO: refactor: CNNNetwork LPT is detected
    InferenceEngine::details::LayerTransformation::Params,
    LayerTestsUtils::LayerTransformation::LptVersion> LayerTransformationParams;

inline std::ostream& operator << (std::ostream &os, const LayerTransformation::LptVersion& value) {
    if ((value != LayerTransformation::LptVersion::cnnNetwork) && (value != LayerTransformation::LptVersion::nGraph)) {
        THROW_IE_EXCEPTION << "unexpected LPT version value " << value;
    }

    os << (value == LayerTransformation::LptVersion::cnnNetwork ? "cnnNetwork" : "nGraph");
    return os;
}

}  // namespace LayerTestsUtils
