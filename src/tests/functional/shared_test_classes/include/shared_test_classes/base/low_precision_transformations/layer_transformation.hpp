// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ov_ops/type_relaxed.hpp"

#include "low_precision/layer_transformation.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

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

    static std::pair<float, float> get_quantization_interval(ov::element::Type precision);

    static std::string to_string(const ov::pass::low_precision::LayerTransformation::Params& params);

    static std::string get_test_case_name_by_params(
        ov::element::Type precision,
        const ov::PartialShape& inputShapes,
        const std::string& targetDevice,
        const ov::pass::low_precision::LayerTransformation::Params& params);

    // get runtime precision by operation friendly name
    std::string get_runtime_precision(const std::string& layerName);

    // get runtime precision by operation type
    std::string get_runtime_precision_by_type(const std::string& layerType);

    std::string get_property_by_type(const std::string& layerTypeName, const std::string& propertyName);

    // get runtime precision by operation friendly name which can be fused
    std::string get_runtime_precision_by_fused_name(const std::string& layerName);

    // check operation sequence in an execution graph and orderedOpsTypes
    // orderedOpsTypes can consist only necessary operations (fewer than exist in the execution graph)
    bool check_execution_order(const std::vector<std::string>& orderedOpsTypes);

    std::map<std::string, ov::Node::RTMap> get_runtime_info();

    void init_input_shapes(const ov::PartialShape& shape);

    void init_input_shapes(const std::vector<ov::PartialShape>& shapes);
};

typedef std::tuple<
    ov::element::Type,
    ov::Shape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

}  // namespace LayerTestsUtils
