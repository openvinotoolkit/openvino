// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/disable_convert_on_const_path.hpp"

#include <memory>
#include <queue>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/variant.hpp>
#include "transformations/rt_info/dequantization_attribute.hpp"

using namespace ngraph;

// TODO: copy/paste from NetworkHelper::isConstantPath
bool isConstantPath(const std::shared_ptr<Node>& op) {
    const auto isNotConstantPathOperation = [](const std::shared_ptr<Node>& node) -> bool {
        return is_type<opset1::Parameter>(node) ||
            is_type<opset1::Convolution>(node) ||
            is_type<opset1::GroupConvolution>(node) ||
            is_type<opset1::MatMul>(node) ||
            is_type<opset1::FakeQuantize>(node);
    };

    if (isNotConstantPathOperation(op)) {
        return false;
    }

    std::queue<Input<Node>> inputs;
    const std::vector<Input<Node>> nodeInputs = op->inputs();
    for (const Input<Node>& nodeInput : nodeInputs) {
        inputs.push(nodeInput);
    }

    while (!inputs.empty()) {
        Input<Node> input = inputs.front();
        inputs.pop();

        const Output<Node>& sourceOutput = input.get_source_output();
        const auto parentNode = sourceOutput.get_node_shared_ptr();
        if (isNotConstantPathOperation(parentNode)) {
            return false;
        }

        for (size_t inputIndex = 0; inputIndex < parentNode->get_input_size(); ++inputIndex) {
            inputs.push(parentNode->input(inputIndex));
        }
    }
    return true;
}

bool canChildBeHandledInLowPrecision(Node* convert) {
    std::queue<Output<Node>> outputs;
    outputs.push(convert->output(0));

    bool allBranchesAreParameters = true;

    while (!outputs.empty()) {
        Output<Node> output = outputs.front();
        outputs.pop();

        const std::set<Input<Node>>& targetInputs = output.get_target_inputs();
        for (const Input<Node>& targetInput : targetInputs) {
            const auto targetNode = targetInput.get_node()->shared_from_this();
            if (is_type<opset1::FakeQuantize>(targetNode)) {
                break;
            }

            if (!isConstantPath(targetNode)) {
                return true;
            }

            for (size_t outputIndex = 0; outputIndex < targetNode->get_output_size(); ++outputIndex) {
                outputs.push(targetNode->output(outputIndex));
            }
        }

        if (outputs.size() == 0) {
            allBranchesAreParameters = false;
        }
    }
    return true;
}


NGRAPH_RTTI_DEFINITION(ngraph::pass::DisableConvertOnConstPath, "DisableConvertOnConstPath", 0);

ngraph::pass::DisableConvertOnConstPath::DisableConvertOnConstPath() {
    auto matcherData = ngraph::pattern::any_input();
    auto matcherConvert = ngraph::pattern::wrap_type<opset3::Convert>({ matcherData }, pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        const auto convert = opsMap.find(matcherConvert)->second.get_node()->shared_from_this();

        // TODO: LPT: move to matcher
        const auto inputPrecision = convert->input(0).get_element_type();
        if ((inputPrecision != element::i8) && (inputPrecision != element::u8)) {
            return false;
        }

        auto parent = convert->get_input_node_ptr(0);
        auto child = convert->output(0).get_target_inputs().begin()->get_node();
        if (is_type<ngraph::opset1::Constant>(parent) && is_type<ngraph::opset1::Subtract>(child) && isConstantPath(convert)) {
            auto& rtInfo = convert->get_rt_info();
            rtInfo["DISABLED_CONSTANT_FOLDING"] = std::make_shared<VariantWrapper<std::string>>("");
            return true;
        }

        ////if (convert->get_friendly_name() == "Convert_287") {
        ////    std::cout << convert->get_friendly_name() << std::endl;
        ////}

        //if (isConstantPath(convert) && canChildBeHandledInLowPrecision(convert)) {
        //    auto& rtInfo = convert->get_rt_info();
        //    rtInfo["DISABLED_CONSTANT_FOLDING"] = std::make_shared<VariantWrapper<std::string>>("");
        //    std::cout << "Constant folding is disabled: " << convert->get_friendly_name() << std::endl;
        //} else {
        //    std::cout << "Constant folding is enabled: " << convert->get_friendly_name() << std::endl;
        //}
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcherConvert, "DisableConvertOnConstPath");
    this->register_matcher(m, callback);
}
