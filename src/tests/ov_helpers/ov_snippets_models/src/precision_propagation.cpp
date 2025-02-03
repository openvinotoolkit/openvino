// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_propagation.hpp"
#include <assert.h>
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> PrecisionPropagationAddFunction::get(
    const ov::element::Type& precision1,
    const ov::PartialShape& inputShape1,
    const ov::element::Type& precision2,
    const ov::PartialShape& inputShape2,
    const ov::element::Type& constant_precision,
    const std::pair<element::Type, element::Type>& convertion_before_op1,
    const element::Type& convertion_before_op2_1,
    const std::pair<element::Type, element::Type>& convertion_before_op2_2,
    const element::Type& convertion_after_op2,
    const element::Type& convertion_before_result) {
    const auto create_convert = [](std::shared_ptr<Node> parent,
                                   const element::Type convertion_type) -> std::shared_ptr<Node> {
        return convertion_type == element::dynamic
                   ? std::dynamic_pointer_cast<Node>(parent)
                   : std::make_shared<ov::snippets::op::ConvertSaturation>(parent, convertion_type);
    };

    const auto make_branch = [&create_convert](
        const ov::element::Type precision,
        const ov::PartialShape& inputShape,
        const size_t index,
        const element::Type convertion_type) -> std::pair<std::shared_ptr<ov::opset1::Parameter>, std::shared_ptr<ov::Node>> {
            const auto parameter = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
            parameter->set_friendly_name("parameter" + std::to_string(index));

            std::shared_ptr<Node> parent = create_convert(parameter, convertion_type);

            return { parameter, parent };
    };

    const auto branch1 = make_branch(precision1, inputShape1, 1, convertion_before_op1.first);
    const auto branch2 = make_branch(precision2, inputShape2, 2, convertion_before_op1.second);

    std::shared_ptr<Node> parent = std::make_shared<DummyAdd>(branch1.second, branch2.second);
    parent->set_friendly_name("add");

    parent = create_convert(parent, convertion_before_op2_1);

    const auto maximum_in2_type =
        convertion_before_op2_2.second == element::dynamic ? constant_precision : convertion_before_op2_2.second;
    if ((convertion_before_op2_2.first == element::dynamic) &&
        (parent->get_output_element_type(0) != maximum_in2_type)) {
        parent = std::make_shared<ov::snippets::op::ConvertSaturation>(parent, maximum_in2_type);
    }

    parent = std::make_shared<ov::opset1::Maximum>(
        create_convert(parent, convertion_before_op2_2.first),
        create_convert(
            std::make_shared<ov::opset1::Constant>(constant_precision, Shape{}, std::vector<float>{0.f}),
            convertion_before_op2_2.second));
    parent->set_friendly_name("maximum");

    parent = create_convert(parent, convertion_after_op2);

    parent = create_convert(parent, convertion_before_result);

    const auto result = std::make_shared<ov::opset1::Result>(parent);
    auto& result_out_tensor = result->get_output_tensor(0);
    result_out_tensor.set_names({ "result_tensor" });
    result->set_friendly_name("result");

    const ov::ResultVector results{ result };
    const ov::ParameterVector parameters{ branch1.first, branch2.first };
    const auto model = std::make_shared<ov::Model>(results, parameters, "SnippetsPrecisionPropagation");
    return model;
}

std::shared_ptr<ov::Model> PrecisionPropagationAddFunction::initOriginal() const {
    return get(
        precision1,
        input_shapes[0],
        precision2,
        input_shapes[1],
        constant_precision,
        actual.convertion_before_op1,
        actual.convertion_before_op2_1,
        actual.convertion_before_op2_2,
        actual.convertion_after_op2);
}

std::shared_ptr<ov::Model> PrecisionPropagationAddFunction::initReference() const {
    return get(
        precision1,
        input_shapes[0],
        precision2,
        input_shapes[1],
        constant_precision,
        expected.convertion_before_op1,
        expected.convertion_before_op2_1,
        expected.convertion_before_op2_2,
        expected.convertion_after_op2,
        expected.convertion_before_result);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
