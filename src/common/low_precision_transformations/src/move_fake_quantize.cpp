// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/move_fake_quantize.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/or.hpp>

#include "low_precision/concat.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MoveFakeQuantize, "MoveFakeQuantize", 0);

MoveFakeQuantize::MoveFakeQuantize(const Params& params) : LayerTransformation(params) {
    const auto concat = ngraph::pattern::wrap_type<opset1::Concat>(pattern::consumers_count(1));
    const auto operation = ngraph::pattern::wrap_type<opset1::Relu>({ concat });
    const auto input_low = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto input_high = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto output_low = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto output_high = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto fq_with_operation = ngraph::pattern::wrap_type<opset1::FakeQuantize>({ operation,
        input_low,
        input_high,
        output_low,
        output_high});
    const auto fq = ngraph::pattern::wrap_type<opset1::FakeQuantize>({ concat,
        input_low,
        input_high,
        output_low,
        output_high });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }

        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(
        std::make_shared<pattern::op::Or>(OutputVector{fq, fq_with_operation}),
        "MoveFakeQuantize");
    this->register_matcher(m, callback);
}

bool MoveFakeQuantize::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    auto fq = m.get_match_root();
    if (!canBeTransformed(context, fq)) {
        return false;
    }
    auto operation = fq->get_input_node_shared_ptr(0);
    std::shared_ptr<ngraph::Node> concat;
    bool without_operation = true;
    const std::string fq_original_name = fq->get_friendly_name();
    std::string operation_original_name;
    if (is_type<opset1::Concat>(operation)) {
        concat = operation;
    } else {
        operation_original_name = operation->get_friendly_name();
        concat = operation->get_input_node_shared_ptr(0);
        without_operation = false;
    }
    if (!ConcatTransformation::isQuantizedStatic(concat)) {
        return false;
    }
    const auto convert_q = fq->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
    bool q_dq = is_type<opset1::Convert>(convert_q);
    std::vector<std::shared_ptr<opset1::Constant>> currConstants(4);
    bool multi_chanels = false;
    const auto number_of_concat_inputs = concat->get_input_size();
    const auto concatNode = as_type_ptr<opset1::Concat>(concat);
    const auto concat_axis = concatNode->get_concatenation_axis();
    for (size_t i = 0; i < 4; i++) {
        currConstants[i] = as_type_ptr<opset1::Constant>(fq->get_input_node_shared_ptr(i + 1));
        if (!multi_chanels && currConstants[i]->get_shape().size() > (concat_axis + 1ul) && currConstants[i]->get_shape()[concat_axis] != 1) {
            multi_chanels = true;
        }
    }
    std::vector<std::vector<std::shared_ptr<ngraph::opset1::Constant>>> newConstants;
    if (multi_chanels) {
        newConstants = NetworkHelper::splitConstantsBeforeConcat(concat, currConstants);
    }
    std::vector<std::shared_ptr<ngraph::Node>> newNodes;
    for (size_t i{ 0 }; i < number_of_concat_inputs; ++i) {
        std::shared_ptr<ngraph::Node> fq_input;
        if (without_operation) {
            fq_input = concat->get_input_node_shared_ptr(i);
        } else {
            auto input = concat->get_input_node_shared_ptr(i);
            fq_input = operation->clone_with_new_inputs({ input });
            fq_input->set_friendly_name(operation_original_name + "_" + std::to_string(i + 1));
        }
        std::shared_ptr<ngraph::Node> newFq;
        if (multi_chanels) {
            newFq = fq->clone_with_new_inputs({ fq_input,
                newConstants[0][newConstants[0].size() == 1 ? 0 : i],
                newConstants[1][newConstants[1].size() == 1 ? 0 : i],
                newConstants[2][newConstants[2].size() == 1 ? 0 : i],
                newConstants[3][newConstants[3].size() == 1 ? 0 : i] });
        } else {
            newFq = fq->clone_with_new_inputs({ fq_input,
                fq->get_input_node_ptr(1)->clone_with_new_inputs({}),
                fq->get_input_node_ptr(2)->clone_with_new_inputs({}),
                fq->get_input_node_ptr(3)->clone_with_new_inputs({}),
                fq->get_input_node_ptr(4)->clone_with_new_inputs({}) });
        }
        ngraph::copy_runtime_info(fq, newFq);
        newFq->set_friendly_name(fq_original_name + "_" + std::to_string(i + 1));
        if (q_dq) {
            auto newConvert_q = convert_q->clone_with_new_inputs({ newFq });
            ngraph::copy_runtime_info(convert_q, newConvert_q);
            newConvert_q->set_friendly_name(convert_q->get_friendly_name() + "_" + std::to_string(i + 1));
            newNodes.push_back(newConvert_q);
        } else {
            newNodes.push_back(newFq);
        }
    }
    auto newConcat = concat->clone_with_new_inputs(ngraph::OutputVector(newNodes.begin(), newNodes.end()));

    newConcat->set_friendly_name(concat->get_friendly_name());
    NetworkHelper::copyInfo(concat, newConcat);
    if (q_dq) {
        auto dq = NetworkHelper::getDequantizationBelow(convert_q);
        moveDequantizationBefore(context, newConcat, dq, false);
        return true;
    }
    replace_node(fq, newConcat);
    updateOutput(context, newConcat, fq);
    return true;
}

bool MoveFakeQuantize::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    auto operation = layer->get_input_node_shared_ptr(0);
    std::shared_ptr<ngraph::Node> concat;
    if (is_type<opset1::Concat>(operation)) {
        concat = operation;
    } else {
        concat = operation->get_input_node_shared_ptr(0);
    }
    if (!ConcatTransformation::isQuantizedStatic(concat)) {
        return false;
    }
    const auto convert_q_target_inputs = layer->output(0).get_target_inputs();
    if (convert_q_target_inputs.empty()) {
        return false;
    }
    const auto convert_q = convert_q_target_inputs.begin()->get_node()->shared_from_this();
    bool q_dq = is_type<opset1::Convert>(convert_q);
    if (q_dq && (convert_q->get_output_size() != 1 || layer->get_output_size() != 1)) {
        return false;
    }
    return true;
}

bool MoveFakeQuantize::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
