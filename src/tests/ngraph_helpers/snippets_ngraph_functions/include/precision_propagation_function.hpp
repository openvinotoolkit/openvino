// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph/opsets/opset1.hpp"
#include "snippets/op/convert_saturation.hpp"

namespace ov {
namespace test {
namespace snippets {

class PrecisionPropagationFunction {
public:
    //
    // Parameter    Parameter
    //    \            /
    //     \          /
    //   Operation #1 with two inputs: 1) data path 2) data path (Add)
    //        \
    //         \    Constant
    //          \    /
    //   Operation #2 with two inputs: 1) data path 2) constant (Maximum)
    //            |
    //            |
    //          Result
    //
    template<typename T>
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision1,
        const ngraph::PartialShape& inputShape1,
        const ngraph::element::Type precision2,
        const ngraph::PartialShape& inputShape2,
        const ngraph::element::Type constant_precision,
        const std::pair<element::Type, element::Type>& convertion_before_op1 = std::pair<element::Type, element::Type>(),
        const std::pair<element::Type, element::Type>& convertion_before_op2 = std::pair<element::Type, element::Type>(),
        const element::Type convertion_after_op2 = {}) {
        const auto branch1 = make_branch(precision1, inputShape1, 1, convertion_before_op1.first);
        const auto branch2 = make_branch(precision2, inputShape2, 2, convertion_before_op1.second);

        std::shared_ptr<Node> parent = std::make_shared<T>(branch1.second, branch2.second);
        parent->set_friendly_name("add");

        const auto maximum_in2_type = convertion_before_op2.second == element::undefined ?
            constant_precision :
            convertion_before_op2.second;
        if ((convertion_before_op2.first == element::undefined) &&
            (parent->get_output_element_type(0) != maximum_in2_type)) {
            parent = std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent, maximum_in2_type);
        }

        parent = std::make_shared<ngraph::opset1::Maximum>(
            create_convert(parent, convertion_before_op2.first),
            create_convert(
                std::make_shared<ngraph::opset1::Constant>(constant_precision, Shape{}, std::vector<float>{0.f}),
                convertion_before_op2.second));
        parent->set_friendly_name("maximum");

        parent = create_convert(parent, convertion_after_op2);

        const auto result = std::make_shared<ngraph::opset1::Result>(parent);
        result->set_friendly_name("result");

        const ngraph::ResultVector results{result};
        const ngraph::ParameterVector parameters{branch1.first, branch2.first};
        const auto model = std::make_shared<ngraph::Function>(results, parameters, "SnippetsPrecisionPropagation");
        model->validate_nodes_and_infer_types();
        return model;
    }

private:
    static std::shared_ptr<Node> create_convert(std::shared_ptr<Node> parent, const element::Type convertion_type) {
        return convertion_type == element::undefined
                   ? std::dynamic_pointer_cast<Node>(parent)
                   : std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent, convertion_type);
    }

    static std::pair<std::shared_ptr<ngraph::opset1::Parameter>, std::shared_ptr<ov::Node>> make_branch(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const size_t index,
        const element::Type convertion_type) {
        const auto parameter = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        parameter->set_friendly_name("parameter" + std::to_string(index));

        std::shared_ptr<Node> parent = create_convert(parameter, convertion_type);

        return {parameter, parent};
    }
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
