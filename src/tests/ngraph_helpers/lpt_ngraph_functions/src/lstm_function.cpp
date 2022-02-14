// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/lstm_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"

#include "ngraph_functions/builders.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> LSTMFunction::get(
    const ngraph::element::Type inputPrecision,
    const std::vector<ngraph::PartialShape>& inputShapes,
    const std::vector<FakeQuantizeOnDataWithConstant>& fqOnDatas,
    const std::vector<DequantizationOperations::Convert>& converts,
    const std::vector<DequantizationOperations>& dequantizations,
    const std::vector<ov::Any>& concatAttributes,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter) {
    auto X = std::make_shared<opset1::Parameter>(inputPrecision, inputShapes[0]);
    X->set_friendly_name("X");
    std::shared_ptr<Node> parent_X = makeQuantizationAndDequantization(X,
                                                                       inputPrecision,
                                                                       X->get_friendly_name(),
                                                                       fqOnDatas[0],
                                                                       converts[0],
                                                                       dequantizations[0]);
    auto H = std::make_shared<opset1::Parameter>(inputPrecision, inputShapes[1]);
    H->set_friendly_name("H");
    std::shared_ptr<Node> parent_H = makeQuantizationAndDequantization(H,
                                                                       inputPrecision,
                                                                       H->get_friendly_name(),
                                                                       fqOnDatas.size() > 1 ? fqOnDatas[1] : fqOnDatas[0],
                                                                       converts.size() > 1 ? converts[1] : converts[0],
                                                                       dequantizations.size() > 1 ? dequantizations[1] : dequantizations[0]);
    auto C = std::make_shared<opset1::Parameter>(inputPrecision, inputShapes[2]);
    C->set_friendly_name("C");
    std::shared_ptr<Node> parent_C = makeQuantizationAndDequantization(C,
                                          inputPrecision,
                                          C->get_friendly_name(),
                                          fqOnDatas.size() > 1 ? fqOnDatas[2] : fqOnDatas[0],
                                          converts.size() > 1 ? converts[2] : converts[0],
                                          dequantizations.size() > 1 ? dequantizations[2] : dequantizations[0]);
    auto w_val = std::vector<float>(512 * 16, 0);
    auto r_val = std::vector<float>(512 * 128, 0);
    auto W = ngraph::opset1::Constant::create(inputPrecision, ngraph::Shape{512, 16}, w_val);
    std::shared_ptr<Node> parent_W = makeQuantizationAndDequantization(W,
                                                                       inputPrecision,
                                                                       W->get_friendly_name(),
                                          fqOnDatas.size() > 1 ? fqOnDatas[3] : fqOnDatas[0],
                                          converts.size() > 1 ? converts[3] : converts[0],
                                          dequantizations.size() > 1 ? dequantizations[3] : dequantizations[0]);
    auto R = ngraph::opset1::Constant::create(inputPrecision, ngraph::Shape{512, 128}, r_val);
    std::shared_ptr<Node> parent_R = makeQuantizationAndDequantization(R,
                                                                       inputPrecision,
                                                                       R->get_friendly_name(),
                                          fqOnDatas.size() > 1 ? fqOnDatas[4] : fqOnDatas[0],
                                          converts.size() > 1 ? converts[4] : converts[0],
                                          dequantizations.size() > 1 ? dequantizations[4] : dequantizations[0]);

    std::shared_ptr<opset1::LSTMCell> lstm;
    if (fqOnDatas.size() > 5) {
        auto b_val = std::vector<float>(512, 0);
        auto B = ngraph::opset6::Constant::create(inputPrecision, ngraph::Shape{512}, b_val);
        std::shared_ptr<Node> parent_B = makeQuantizationAndDequantization(B,
                                                                           inputPrecision,
                                                                           B->get_friendly_name(),
                                              fqOnDatas.size() > 1 ? fqOnDatas[5] : fqOnDatas[0],
                                              converts.size() > 1 ? converts[5] : converts[0],
                                              dequantizations.size() > 1 ? dequantizations[5] : dequantizations[0]);
        lstm = std::make_shared<opset1::LSTMCell>(parent_X, parent_H, parent_C, parent_W, parent_R, parent_B, 128);
    }
    lstm = std::make_shared<opset1::LSTMCell>(parent_X, parent_H, parent_C, parent_W, parent_R, 128);
    lstm->set_friendly_name("lstm");

    auto& rtInfo = lstm->get_rt_info();
    rtInfo["Variant::std::string"] = "lstm";

    const auto lastDequantization = makeDequantization(lstm, dequantizationAfter);
    std::shared_ptr<ngraph::Node> parent = lastDequantization;
    parent->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{X, H, C},
        "LSTMTransformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
