// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/move_fake_quantize.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

MoveFakeQuantize::MoveFakeQuantize(const Params& params) : LayerTransformation(params) {
    auto matcher = ngraph::pattern::wrap_type<opset1::FakeQuantize>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
}

        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "MoveFakeQuantize");
    this->register_matcher(m, callback);
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

bool MoveFakeQuantize::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    auto fq = m.get_match_root();
    auto relu = fq->get_input_node_shared_ptr(0);
    auto concat = relu->get_input_node_shared_ptr(0);
    auto result = *fq->output(0).get_target_inputs().begin();
    auto input1 = concat->get_input_node_shared_ptr(0);
    auto input2 = concat->get_input_node_shared_ptr(1);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(input1->output(0));
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(input2->output(0));
    auto fq1 = std::make_shared<opset1::FakeQuantize>(relu1,
        fq->get_input_node_shared_ptr(1),
        fq->get_input_node_shared_ptr(2),
        fq->get_input_node_shared_ptr(3),
        fq->get_input_node_shared_ptr(4),
        as_type_ptr<opset1::FakeQuantize>(fq)->get_levels());
    auto fq2 = std::make_shared<opset1::FakeQuantize>(relu2,
        fq->get_input_node_shared_ptr(1),
        fq->get_input_node_shared_ptr(2),
        fq->get_input_node_shared_ptr(3),
        fq->get_input_node_shared_ptr(4),
        as_type_ptr<opset1::FakeQuantize>(fq)->get_levels());

    auto new_concat = concat->clone_with_new_inputs({ fq1->output(0), fq2->output(0) });
    auto& rtInfo = new_concat->get_rt_info();
    new_concat->set_friendly_name("output");

    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");
    replace_node(concat, new_concat);
    replace_node(fq, new_concat);

    /*insert_new_node_between(input1, concat, fq1);
    insert_new_node_between(input1, fq1, relu1);
    insert_new_node_between(input2, concat, fq2);
    insert_new_node_between(input2, fq2, relu2);
    result.replace_source_output(concat->output(0));*/

    return true;
}

bool MoveFakeQuantize::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
