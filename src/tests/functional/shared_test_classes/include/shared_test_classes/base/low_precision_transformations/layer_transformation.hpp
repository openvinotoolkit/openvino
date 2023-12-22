// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "low_precision/layer_transformation.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsUtils {

class LayerTransformationParamsNGraphFactory {
public:
    static ov::pass::low_precision::LayerTransformation::Params createParamsU8I8AndI8();
    static ov::pass::low_precision::LayerTransformation::Params createParamsU8I8();
    static ov::pass::low_precision::LayerTransformation::Params createParamsI8I8();
    static ov::pass::low_precision::LayerTransformation::Params createParams();
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

    static std::pair<float, float> getQuantizationInterval(const ngraph::element::Type precision);

    static std::string toString(const ov::pass::low_precision::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const InferenceEngine::Precision precision,
        const InferenceEngine::SizeVector& inputShapes,
        const std::string& targetDevice,
        const ov::pass::low_precision::LayerTransformation::Params& params);

    static std::string getTestCaseNameByParams(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShapes,
        const std::string& targetDevice,
        const ov::pass::low_precision::LayerTransformation::Params& params);
};

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

}  // namespace LayerTestsUtils
