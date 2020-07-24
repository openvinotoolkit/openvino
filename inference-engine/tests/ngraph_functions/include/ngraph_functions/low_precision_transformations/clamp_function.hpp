// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <vector>
#include <ngraph/ngraph.hpp>
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ClampFunction {
public:
    class ActualValues {
    public:
        ngraph::element::Type lowPrecision;
        std::vector<float> subtractValues;
        std::vector<float> multiplyValues;
    };

    class ExpectedValues {
    public:
        ngraph::element::Type lowPrecision;
        std::vector<float> subtractValues;
        std::vector<float> multiplyValues;
    };

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const bool updatePrecisions,
        const ActualValues& values);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
        const double clampLowConst,
        const double clampHighConst);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const bool updatePrecisions,
        const ExpectedValues& values);
};

inline std::ostream& operator<<(std::ostream& out, const ClampFunction::ActualValues& values) {
    std::ostringstream result;
    result << "_" << values.lowPrecision << "_subtract_[_";
    for (const auto& value : values.subtractValues) {
        result << value << "_";
    }
    result << "]_multiply_[_";
    for (const auto& value : values.multiplyValues) {
        result << value << "_";
    }
    result << "]";

    return out << result.str();
}

inline std::ostream& operator<<(std::ostream& out, const ClampFunction::ExpectedValues& values) {
    std::ostringstream result;
    result << "_" << values.lowPrecision << "_subtract_[_";
    for (const auto& value : values.subtractValues) {
        result << value << "_";
    }
    result << "]_multiply_[_";
    for (const auto& value : values.multiplyValues) {
        result << value << "_";
    }
    result << "]";

    return out << result.str();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
