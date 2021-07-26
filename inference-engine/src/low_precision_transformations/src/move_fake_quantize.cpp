// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/move_fake_quantize.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "low_precision/common/subgraph.hpp"

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

MoveFakeQuantize::MoveFakeQuantize(const Params& params) : LayerTransformation(params) {
}

void MoveFakeQuantize::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    /*addPattern(
        pass,
        context,
        make_op_pattern<opset1::FakeQuantize>({ make_op_label<opset1::Concat>() }));*/
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

bool MoveFakeQuantize::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op);
    if (dequantization.empty()) {
        return false;
    }

    const std::vector<float> scales = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1))->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.0; })) {
        return false;
    }

    return true;
}

bool MoveFakeQuantize::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    auto fq = m.get_match_root();
    auto concat = fq->get_input_node_shared_ptr(0);
    auto result = *fq->output(0).get_target_inputs().begin();
    
    auto input1 = concat->get_input_node_shared_ptr(0);
    auto input2 = concat->get_input_node_shared_ptr(1);
    auto fq1 = std::make_shared<opset1::FakeQuantize>(input1,
        fq->get_input_node_shared_ptr(1),
        fq->get_input_node_shared_ptr(2),
        fq->get_input_node_shared_ptr(3),
        fq->get_input_node_shared_ptr(4),
        255);
    auto fq2 = std::make_shared<opset1::FakeQuantize>(input2,
        fq->get_input_node_shared_ptr(1),
        fq->get_input_node_shared_ptr(2),
        fq->get_input_node_shared_ptr(3),
        fq->get_input_node_shared_ptr(4),
        255);
    std::vector<Input<Node>> dst_inputs1 = get_inputs_from(*input1, *concat),
                             dst_inputs2 = get_inputs_from(*input2, *concat);
    auto& dst_input1 = dst_inputs1[0],
          dst_input2 = dst_inputs2[0];

    std::vector<Output<Node>> src_outputs1 = get_outputs_to(*input1, *concat),
                              src_outputs2 = get_outputs_to(*input2, *concat);
    auto& src_output1 = src_outputs1[0],
          src_output2 = src_outputs2[0];

    src_output1.remove_target_input(dst_input1);
    src_output2.remove_target_input(dst_input2);
    dst_input1.replace_source_output(
        fq1->output(0));
    dst_input2.replace_source_output(
        fq2->output(0));
    std::vector<Output<Node>> src_outputs_concat = get_outputs_to(*concat, *fq);
    std::vector<Input<Node>> dst_inputs_FQ = get_inputs_from(*concat, *fq);
    auto& src_output_concat = src_outputs_concat[0];
    auto& dst_input_FQ = dst_inputs_FQ[0];
    src_output_concat.remove_target_input(dst_input_FQ);
    result.replace_source_output(
        concat->output(0));
    return true;
}

bool MoveFakeQuantize::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
