// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "two_binary_ops_function.hpp"
#include <ngraph/opsets/opset1.hpp>
#include "snippets/op/convert_saturation.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ngraph::Function> TwoBinaryOpsFunction::get(
    const ngraph::element::Type& precision1,
    const ngraph::PartialShape& inputShape1,
    const ngraph::element::Type& precision2,
    const ngraph::PartialShape& inputShape2,
    const ngraph::element::Type& constant_precision,
    const std::pair<element::Type, element::Type>& convertion_before_op1,
    const element::Type& convertion_before_op2_1,
    const std::pair<element::Type, element::Type>& convertion_before_op2_2,
    const std::vector<Branch>& convertion_after_op2,
    const element::Type& convertion_before_result,
    const std::map<std::vector<element::Type>, std::vector<element::Type>>& supported_out_precisions1,
    const std::map<std::vector<element::Type>, std::vector<element::Type>>& supported_out_precisions2) {
    const auto create_convert = [](
        const std::shared_ptr<Node>& parent,
        const element::Type& convertion_type,
        const std::string& name) -> std::shared_ptr<Node> {
        if (convertion_type == element::undefined) {
            return parent;
        }
        auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(parent, convertion_type);
        convert->set_friendly_name(name);
        return convert;
    };

    ngraph::ResultVector results;

    const auto make_branch = [&create_convert](
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape,
        const size_t index,
        const element::Type convertion_type) -> std::pair<std::shared_ptr<ngraph::opset1::Parameter>, std::shared_ptr<ov::Node>> {
            const auto parameter = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
            parameter->set_friendly_name("parameter" + std::to_string(index));

            std::shared_ptr<Node> parent = create_convert(parameter, convertion_type, "convert_before1_" + std::to_string(index));

            return { parameter, parent };
    };

    const auto branch1 = make_branch(precision1, inputShape1, 1, convertion_before_op1.first);
    const auto branch2 = make_branch(precision2, inputShape2, 2, convertion_before_op1.second);

    std::shared_ptr<Node> parent = std::make_shared<DummyOperation1>(branch1.second, branch2.second, supported_out_precisions1);
    parent->set_friendly_name("operation1");

    parent = create_convert(parent, convertion_before_op2_1, "convert_before2_1");

    parent = std::make_shared<DummyOperation2>(
        create_convert(parent, convertion_before_op2_2.first, "convert_before2_1"),
        create_convert(
            std::make_shared<ngraph::opset1::Constant>(constant_precision, Shape{}, std::vector<float>{0.f}),
            convertion_before_op2_2.second,
            "convert_before2_2"),
        supported_out_precisions2);
    parent->set_friendly_name("operation2");

    auto name_index = 2ull;
    for (auto index = 1ull; index < convertion_after_op2.size(); index++) {
        const auto& branch = convertion_after_op2[index];
        auto parent2 = parent;
        parent2 = create_convert(parent2, branch.type, "convert_after2_branch" + std::to_string(index));

        if (branch.branches_amount > 0) {
            for (auto branch_index = 0ull; branch_index < branch.branches_amount; ++branch_index) {
                const auto relu = std::make_shared<ngraph::opset1::Relu>(parent2);
                relu->set_friendly_name("relu" + std::to_string(name_index));

                const auto result2 = std::make_shared<ngraph::opset1::Result>(relu);
                results.push_back(result2);
                auto& result_out_tensor = result2->get_output_tensor(0);
                result_out_tensor.set_names({ "result_tensor" + std::to_string(name_index) });
                result2->set_friendly_name("result" + std::to_string(name_index));
                ++name_index;
            }
        } else {
            const auto result2 = std::make_shared<ngraph::opset1::Result>(parent2);
            results.push_back(result2);
            auto& result_out_tensor = result2->get_output_tensor(0);
            result_out_tensor.set_names({ "result_tensor" + std::to_string(name_index) });
            result2->set_friendly_name("result" + std::to_string(name_index));
            ++name_index;
        }
    }

    parent = create_convert(
        parent,
        convertion_after_op2.empty() ? element::undefined : convertion_after_op2[0].type,
        "convert_after2");

    parent = create_convert(parent, convertion_before_result, "convert_before_result");

    const auto result = std::make_shared<ngraph::opset1::Result>(parent);
    results.push_back(result);
    auto& result_out_tensor = result->get_output_tensor(0);
    result_out_tensor.set_names({ "result_tensor" });
    result->set_friendly_name("result");

    const ngraph::ParameterVector parameters{ branch1.first, branch2.first };
    const auto model = std::make_shared<ngraph::Function>(results, parameters, "SnippetsPrecisionPropagation");
    return model;
}

std::shared_ptr<ov::Model> TwoBinaryOpsFunction::initOriginal() const {
    return get(
        precision1,
        input_shapes[0],
        precision2,
        input_shapes[1],
        constant_precision,
        actual.convertion_before_op1,
        actual.convertion_before_op2_1,
        actual.convertion_before_op2_2,
        actual.convertion_after_op2,
        {},
        actual.supported_out_precisions1,
        actual.supported_out_precisions2);
}

std::shared_ptr<ov::Model> TwoBinaryOpsFunction::initReference() const {
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
        expected.convertion_before_result,
        actual.supported_out_precisions1,
        actual.supported_out_precisions2);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
