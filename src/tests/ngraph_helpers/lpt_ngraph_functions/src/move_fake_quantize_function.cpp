// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/move_fake_quantize_function.hpp"
#include <low_precision/relu.hpp>

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> MoveFakeQuantize::get(
    const ngraph::element::Type inputPrecision,
    const std::vector<ngraph::PartialShape>& inputShape,
    const size_t number_of_operations,
    const std::vector<FakeQuantizeOnDataWithConstant>& fqOnDataBefore,
    const DequantizationOperations::Convert& convertBefore,
    const DequantizationOperations& dequantizationBefore,
    const std::string& operation,
    const FakeQuantizeOnDataWithConstant& fqOnDataAfter,
    const DequantizationOperations::Convert& convertAfter,
    const DequantizationOperations& dequantizationAfter,
    const std::vector<ov::Any>& concatAttributes,
    const ngraph::element::Type precisionAfterOperation,
    const std::int64_t& axis) {

    std::vector <std::shared_ptr<ngraph::opset1::Parameter>> inputs(number_of_operations);
    std::vector <std::shared_ptr<ngraph::Node>> parents(number_of_operations);
    for (size_t i = 0; i < number_of_operations; i++) {
        auto ind = 0;
        if (inputShape.size() != 1) {
            ind = i;
        }
        inputs[i] = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape[ind]);
        inputs[i]->set_friendly_name(std::string("input") + "_" + std::to_string(i + 1));
        parents[i] = inputs[i];
    }
    if (!fqOnDataBefore.empty()) {
        for (size_t i = 0; i < number_of_operations; i++) {
            size_t ind = i;
            if (fqOnDataBefore.size() == 1) {
                ind = 0;
            }
            if (operation == "relu") {
                auto relu = std::make_shared<ngraph::opset1::Relu>(parents[i]->output(0));
                parents[i] = makeFakeQuantize(relu, inputPrecision, fqOnDataBefore[ind]);
            } else {
                parents[i] = makeFakeQuantize(parents[i], inputPrecision, fqOnDataBefore[ind]);
            }
            parents[i]->set_friendly_name(std::string("concat_fq") + "_" + std::to_string(i + 1));
            if (!convertBefore.empty()) {
                parents[i] = std::make_shared<opset1::Convert>(parents[i], convertBefore.outPrecision);
            }
            if (!dequantizationBefore.empty()) {
                parents[i] = makeDequantization(parents[i], dequantizationBefore);
            }
        }
    }
    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector(parents.begin(), parents.end()), axis);
    concat->set_friendly_name("concat");
    std::shared_ptr<ngraph::Node> parent = concat;
    addAttributes({ parent }, concatAttributes);
    if (!fqOnDataAfter.empty()) {
        std::shared_ptr<ngraph::Node> fq;
        if (operation == "relu") {
            auto relu = std::make_shared<ngraph::opset1::Relu>(concat->output(0));
            fq = makeFakeQuantize(relu, inputPrecision, fqOnDataAfter);
        } else {
            fq = makeFakeQuantize(concat, inputPrecision, fqOnDataAfter);
        }
        fq->set_friendly_name("fakeQuantizeAfter");
        parent = fq;
        if (!convertAfter.empty()) {
            parent = std::make_shared<opset1::Convert>(parent, convertAfter.outPrecision);
        }
        if (!dequantizationAfter.empty()) {
            parent = makeDequantization(parent, dequantizationAfter);
        }
    }
    parent->set_friendly_name("output");
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector(inputs.begin(), inputs.end()),
        "MoveFakeQuantize");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
