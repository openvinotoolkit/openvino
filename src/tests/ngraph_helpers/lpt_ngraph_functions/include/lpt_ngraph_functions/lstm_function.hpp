// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class LSTMFunction {
public:

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type inputPrecision,
        const std::vector<ngraph::PartialShape>& inputShapes,
        const std::vector<FakeQuantizeOnDataWithConstant>& fqOnDatas,
        const std::vector<DequantizationOperations::Convert>& converts,
        const std::vector<DequantizationOperations>& dequantizations,
        const std::vector<ov::Any>& concatAttributes,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter);
};

template <typename T>
std::shared_ptr<Node> makeQuantizationAndDequantization(const std::shared_ptr<T>& input,
                                                        const ngraph::element::Type inputPrecision,
                                                        const std::string friendly_name,
                                                        const FakeQuantizeOnDataWithConstant& fqOnData,
                                                        const DequantizationOperations::Convert& convert,
                                                        const DequantizationOperations& dequantization) {
    std::shared_ptr<Node> parent;
    if (fqOnData.empty()) {
        parent = input;
    } else {
        std::shared_ptr<Node> fakeQuantize1 = makeFakeQuantizeTypeRelaxed(input, inputPrecision, fqOnData);
        fakeQuantize1->set_friendly_name("fakeQuantize_" + friendly_name);
        parent = fakeQuantize1;
    }
    if (!convert.empty()) {
        parent = std::make_shared<opset1::Convert>(parent, convert.outPrecision);
    }
    if (!dequantization.empty()) {
        parent = makeDequantization(parent, dequantization);
    }
    return parent;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
