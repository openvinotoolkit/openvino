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
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

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

class LayerTransformation : virtual public ov::test::SubgraphBaseTest {
protected:
    LayerTransformation();

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

    // get runtime precision by operation friendly name
    std::string getRuntimePrecision(const std::string& layerName);

    // get runtime precision by operation type
    std::string getRuntimePrecisionByType(const std::string& layerType);

    // get runtime precision by operation friendly name which can be fused
    std::string getRuntimePrecisionByFusedName(const std::string& layerName);

    std::map<std::string, ngraph::Node::RTMap> getRuntimeInfo();

    void init_input_shapes(const ov::PartialShape& shape);

    void init_input_shapes(const std::vector<ov::PartialShape>& shapes);
};

typedef std::tuple<
    InferenceEngine::Precision,
    InferenceEngine::SizeVector,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

}  // namespace LayerTestsUtils
