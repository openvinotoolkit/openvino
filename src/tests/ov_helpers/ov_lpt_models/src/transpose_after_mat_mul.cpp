// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/transpose_after_mat_mul.hpp"
#include "low_precision/network_helper.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ov_lpt_models/common/builders.hpp"
#include "ov_models/subgraph_builders.hpp"

using namespace ov::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> TransposeAfterMatMulFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape) {
        const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        input1->set_friendly_name("input1");

        const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        input2->set_friendly_name("input2");

        const float k = 50.f;
        const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(input1, precision, 256ul, { 1ul }, { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
        input2->set_friendly_name("fakeQuantize1");
        const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(input2, precision, 256ul, { 1ul }, { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });
        input2->set_friendly_name("fakeQuantize2");
        const auto matMul = std::make_shared<ngraph::opset1::MatMul>(fakeQuantize1, fakeQuantize2, false, false);
        input2->set_friendly_name("matMul");
        const auto transpose = std::make_shared<ngraph::opset1::Transpose>(
            matMul,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 4ul }, { 0, 2, 1, 3 }));
        transpose->set_friendly_name("transpose");

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(transpose) };
        std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{ input1, input2 },
            "TransposeAfterMatMulFunction");

        return function;
    }
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
