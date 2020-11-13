// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class AddActualValues {
public:
    ngraph::element::Type precision1;
    std::vector<float> subtractValues1;
    std::vector<float> mutliplyValues1;
    ngraph::element::Type precision2;
    std::vector<float> subtractValues2;
    std::vector<float> mutliplyValues2;
};

inline std::ostream& operator<<(std::ostream& out, const AddActualValues& values) {
    return out <<
        "_" << values.precision1 <<
        "_subtract" << values.subtractValues1.size() <<
        "_mutliply" << values.mutliplyValues1.size() <<
        "_" << values.precision2 <<
        "_subtract" << values.subtractValues2.size() <<
        "_mutliply" << values.mutliplyValues2.size();
}

class AddExpectedValues {
public:
    ngraph::element::Type precision1;
    std::vector<float> subtractValues1;
    std::vector<float> mutliplyValues1;
    ngraph::element::Type precision2;
    std::vector<float> mutliplyValuesAfter;
};

inline std::ostream& operator<<(std::ostream& out, const AddExpectedValues& values) {
    return out <<
        "_" << values.precision1 <<
        "_subtract" << values.subtractValues1.size() <<
        "_mutliply" << values.mutliplyValues1.size() <<
        "_" << values.precision2 <<
        "_mutliply" << values.mutliplyValuesAfter.size();
}

class ElementwiseWithMultiParentDequantizationFunction {
public:
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::element::Type& precision1,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
        const ngraph::element::Type& precision2,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization2);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
