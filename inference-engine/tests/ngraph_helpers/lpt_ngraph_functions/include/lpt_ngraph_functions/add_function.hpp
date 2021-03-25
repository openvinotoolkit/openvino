// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <low_precision/layer_transformation.hpp>

#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

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

class AddOperation {
public:
    ngraph::builder::subgraph::Constant constantOnWeights;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnWeights;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations;
    std::string operationType;

    AddOperation() {}

    AddOperation(
        const Constant& constantOnWeights,
        const FakeQuantizeOnData& fakeQuantizeOnWeights,
        const DequantizationOperations& dequantizationOperations,
        const std::string& operationType) :
        constantOnWeights(constantOnWeights),
        fakeQuantizeOnWeights(fakeQuantizeOnWeights),
        dequantizationOperations(dequantizationOperations),
        operationType(operationType)
    {}

    bool empty() const {
        return
            constantOnWeights.empty() &&
            fakeQuantizeOnWeights.empty() &&
            dequantizationOperations.empty() &&
            operationType.empty();
    }
};

inline std::ostream& operator<<(std::ostream& out, const AddOperation& operation) {
    return out << "_" <<
        operation.constantOnWeights << "_" <<
        operation.fakeQuantizeOnWeights << "_" <<
        operation.dequantizationOperations << "_" <<
        operation.operationType;
}

class AddFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool broadcast,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::element::Type& precision1,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
        const ngraph::element::Type& precision2,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
        const int constInput,
        const ngraph::builder::subgraph::Constant constant,
        const std::string& additionalLayer);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool broadcast,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData1,
        const AddOperation& operation1,
        const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData2,
        const AddOperation& operation2,
        const int constInput);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool broadcast,
        const ngraph::pass::low_precision::LayerTransformation::Params& params,
        const ngraph::element::Type& precision1,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
        const ngraph::element::Type& precision2,
        const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
        const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
        const int constInput,
        const ngraph::builder::subgraph::Constant constant,
        const std::string& additionalLayer,
        const std::string& operationType);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
