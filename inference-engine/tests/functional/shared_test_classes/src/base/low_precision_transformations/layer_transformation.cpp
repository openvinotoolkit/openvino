// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"

#include <vector>
#include <string>

#include <ie_core.hpp>
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

using namespace InferenceEngine;
using namespace ngraph;

namespace LayerTestsUtils {
ngraph::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsU8I8AndI8() {
    return ngraph::pass::low_precision::LayerTransformation::Params();
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsU8I8() {
    return ngraph::pass::low_precision::LayerTransformation::Params();
}

ngraph::pass::low_precision::LayerTransformation::Params LayerTransformationParamsNGraphFactory::createParamsI8I8() {
    return ngraph::pass::low_precision::LayerTransformation::Params();
}

LayerTransformation::LayerTransformation() {
    threshold = 0.05;
    auto& configuration = GetConfiguration();
    configuration[PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE] = PluginConfigParams::YES;
}

InferenceEngine::Blob::Ptr LayerTransformation::GenerateInput(
    const ngraph::element::Type precision,
    const InferenceEngine::TensorDesc& tensorDesc,
    const float k) {
    const auto interval = getQuantizationInterval(precision);
    const float low = interval.first / k;
    const float hight = interval.second / k;

    return FuncTestUtils::createAndFillBlobConsistently(tensorDesc, hight - low, static_cast<int32_t>(low), 1ul);
}

std::pair<float, float> LayerTransformation::getQuantizationInterval(const ngraph::element::Type precision) {
    const bool unsignedInterval = precision == ngraph::element::u8;
    const float low = unsignedInterval ? 0.f : -128.f;
    const float hight = unsignedInterval ? 255.f : 127.f;
    return std::make_pair(low, hight);
}

std::string LayerTransformation::toString(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    using namespace ngraph::pass::low_precision;
    std::ostringstream result;
    result <<
        (params.updatePrecisions ? "" : "notUpdatePrecisions_") <<
        params.deqPrecision;

    return result.str();
}

std::string LayerTransformation::getTestCaseNameByParams(
    const InferenceEngine::Precision precision,
    const InferenceEngine::SizeVector& inputShapes,
    const std::string& targetDevice,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result << precision.name() << "_" << ngraph::Shape(inputShapes) << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

std::string LayerTransformation::getTestCaseNameByParams(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShapes,
    const std::string& targetDevice,
    const ngraph::pass::low_precision::LayerTransformation::Params& params) {
    std::ostringstream result;
    result << precision << "_" << inputShapes << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

}  // namespace LayerTestsUtils
