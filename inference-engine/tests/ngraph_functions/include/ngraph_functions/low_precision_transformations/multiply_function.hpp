// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MultiplyActualValues {
public:
    ngraph::element::Type precision1;
    std::vector<float> subtractValues1;
    std::vector<float> mutliplyValues1;
    ngraph::element::Type precision2;
    std::vector<float> subtractValues2;
    std::vector<float> mutliplyValues2;
};

inline std::ostream& operator<<(std::ostream& out, const MultiplyActualValues& values) {
    return out <<
        "_" << values.precision1 <<
        "_subtract" << values.subtractValues1.size() <<
        "_mutliply" << values.mutliplyValues1.size() <<
        "_" << values.precision2 <<
        "_subtract" << values.subtractValues2.size() <<
        "_mutliply" << values.mutliplyValues2.size();
}

class MultiplyExpectedValues {
public:
    ngraph::element::Type precision1;
    std::vector<float> subtractValues1;
    std::vector<float> mutliplyValues1;
    ngraph::element::Type precision2;
    std::vector<float> subtractValues2;
    std::vector<float> mutliplyValues2;
};

inline std::ostream& operator<<(std::ostream& out, const MultiplyExpectedValues& values) {
    return out <<
        "_" << values.precision1 <<
        "_subtract" << values.subtractValues1.size() <<
        "_mutliply" << values.mutliplyValues1.size() <<
        "_" << values.precision2 <<
        "_subtract" << values.subtractValues2.size() <<
        "_mutliply" << values.mutliplyValues2.size();
}

class MultiplyFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool& broadcast,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const MultiplyActualValues& actualValues,
        const bool& constInput);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool& broadcast,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const MultiplyExpectedValues& actualValues,
        const bool& constInput);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
