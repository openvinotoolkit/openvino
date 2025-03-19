// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <low_precision/layer_transformation.hpp>

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class AddActualValues {
public:
    ov::element::Type precision1;
    std::vector<float> subtractValues1;
    std::vector<float> mutliplyValues1;
    ov::element::Type precision2;
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
    ov::element::Type precision1;
    std::vector<float> subtractValues1;
    std::vector<float> mutliplyValues1;
    ov::element::Type precision2;
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
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::pass::low_precision::LayerTransformation::Params& params,
        const ov::element::Type& precision1,
        const ov::builder::subgraph::DequantizationOperations& dequantization1,
        const ov::element::Type& precision2,
        const ov::builder::subgraph::DequantizationOperations& dequantization2);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
