// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/mul_add_to_scaleshift_or_power_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "transformations/low_precision/network_helper.hpp"

#include <legacy/ngraph_ops/power.hpp>
#include <legacy/ngraph_ops/scaleshift.hpp>

#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
    using namespace ngraph::pass;
    std::shared_ptr<ngraph::Function> MulAddToScaleshiftOrPowerFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        bool isDequantization,
        const ngraph::builder::subgraph::DequantizationOperations::Multiply& mulValues,
        const ngraph::builder::subgraph::Add& addValues) {
        const auto input = std::make_shared<ngraph::op::v0::Parameter>(precision, inputShape);

        const auto mulConst = ngraph::op::Constant::create(ngraph::element::f32, mulValues.constantShape, mulValues.values);
        const auto mul = std::make_shared<ngraph::op::TypeRelaxed<ngraph::pass::low_precision::DequantizationMultiply>>(
            std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(input, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(mulConst, element::f32).get());

        const auto addConst = ngraph::op::Constant::create(ngraph::element::f32, addValues.constantShape, addValues.values);
        const auto add = std::make_shared<ngraph::pass::low_precision::DequantizationAdd>(mul, addConst);
        add->set_friendly_name("add");

        if (!isDequantization) {
            ngraph::pass::low_precision::NetworkHelper::cleanRunTimeInfo(mul);
            ngraph::pass::low_precision::NetworkHelper::cleanRunTimeInfo(add);
        }

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MulAddToScaleshiftOrPowerFunction");
    }

    std::shared_ptr<ngraph::Function> MulAddToScaleshiftOrPowerFunction::getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        bool isDequantization,
        const ngraph::builder::subgraph::DequantizationOperations::Multiply& weightsValues,
        const ngraph::builder::subgraph::Add& biasesValues,
        const ngraph::element::Type precisionAfterOperation) {
        const auto input = std::make_shared<ngraph::op::v0::Parameter>(precision, inputShape);

        ngraph::Shape constShape = { 1, inputShape[1], 1, 1 };
        const auto weights = ngraph::op::Constant::create(ngraph::element::f32, constShape, weightsValues.values);
        const auto biases = ngraph::op::Constant::create(ngraph::element::f32, constShape, biasesValues.values);

        std::shared_ptr<ngraph::Node> lastNode;
        if (isDequantization) {
            const auto scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(input, weights, biases, precisionAfterOperation);
            scaleshift->set_friendly_name("add");
            lastNode = scaleshift;
        } else {
            float scale = weightsValues.values[0];
            float shift = biasesValues.values[0];
            const auto power = std::make_shared<ngraph::op::PowerIE>(input, 1.f, scale, shift, precisionAfterOperation);
            power->set_friendly_name("add");
            lastNode = power;
        }


        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastNode) };
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MulAddToScaleshiftOrPowerFunction");
    }
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
