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

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MoveFakeQuantize, "MoveFakeQuantize", 0);

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

bool MoveFakeQuantize::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    auto fq = m.get_match_root();
    auto operation = fq->get_input_node_shared_ptr(0);

    // TODO: temporary to enable other transformations <= update matcher instead this validation
    if (!is_type<opset1::Relu>(operation) && !is_type<opset1::Concat>(operation)) {
        return false;
    }

    auto type = operation->get_type_name();
    std::shared_ptr<ngraph::Node> concat, fq1input, fq2input;
    if (strcmp(type, "Concat") == 0) {
        concat = operation;
        fq1input = operation->get_input_node_shared_ptr(0);
        fq2input = operation->get_input_node_shared_ptr(1);
    } else {
        concat = operation->get_input_node_shared_ptr(0);
        auto input1 = concat->get_input_node_shared_ptr(0);
        auto input2 = concat->get_input_node_shared_ptr(1);
        if (strcmp(type, "Relu") == 0) {
            fq1input = std::make_shared<ngraph::opset1::Relu>(input1->output(0));
            fq2input = std::make_shared<ngraph::opset1::Relu>(input2->output(0));
        }
    }

    auto fq1 = std::make_shared<opset1::FakeQuantize>(fq1input,
        fq->get_input_node_shared_ptr(1),
        fq->get_input_node_shared_ptr(2),
        fq->get_input_node_shared_ptr(3),
        fq->get_input_node_shared_ptr(4),
        as_type_ptr<opset1::FakeQuantize>(fq)->get_levels());
    auto fq2 = std::make_shared<opset1::FakeQuantize>(fq2input,
        fq->get_input_node_shared_ptr(1),
        fq->get_input_node_shared_ptr(2),
        fq->get_input_node_shared_ptr(3),
        fq->get_input_node_shared_ptr(4),
        as_type_ptr<opset1::FakeQuantize>(fq)->get_levels());

    auto new_concat = concat->clone_with_new_inputs({ fq1->output(0), fq2->output(0) });

    replace_node(concat, new_concat);
    //NetworkHelper::copyInfo(new_concat, fq);
    replace_node(fq, new_concat);
    updateOutput(context, new_concat, fq);
    return true;
}

bool MoveFakeQuantize::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
