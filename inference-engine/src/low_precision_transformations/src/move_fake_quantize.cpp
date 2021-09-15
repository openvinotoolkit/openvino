// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/move_fake_quantize.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <memory>
#include <ngraph/ngraph.hpp>
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
    auto operation = fq->get_input_node_shared_ptr(0);
    std::shared_ptr<ngraph::Node> concat;
    bool only_concat = true;
    std::string fq_original_name = fq->get_friendly_name(), operation_original_name;
    if (is_type<opset1::Concat>(operation)) {
        concat = operation;
    } else {
        operation_original_name = operation->get_friendly_name();
        concat = operation->get_input_node_shared_ptr(0);
        only_concat = false;
    }
    if (!ConcatTransformation::isQuantizedStatic(concat)) {
        return false;
    }
    std::vector<std::shared_ptr<ngraph::Node>> fqs;
    size_t input_size = concat->get_input_size();
    for (size_t i{ 0 }; i < input_size; ++i) {
        std::shared_ptr<ngraph::Node> fq_input;
        if (only_concat) {
            fq_input = concat->get_input_node_shared_ptr(i);
        } else {
            auto input = concat->get_input_node_shared_ptr(i);
            fq_input = operation->clone_with_new_inputs({ input });
            fq_input->set_friendly_name(operation_original_name + "_" + std::to_string(i + 1));
        }
        auto newFq = fq->clone_with_new_inputs({ fq_input,
            fq->get_input_node_shared_ptr(1),
            fq->get_input_node_shared_ptr(2),
            fq->get_input_node_shared_ptr(3),
            fq->get_input_node_shared_ptr(4) });
        newFq->set_friendly_name(fq_original_name + "_" + std::to_string(i + 1));
        fqs.push_back(newFq);
    }
    ngraph::copy_runtime_info(fq, fqs);
    auto newConcat = concat->clone_with_new_inputs(ngraph::OutputVector(fqs.begin(), fqs.end()));
    newConcat->set_friendly_name(concat->get_friendly_name());
    replace_node(fq, newConcat);
    NetworkHelper::copyInfo(concat, newConcat);
    updateOutput(context, newConcat, fq);
    return true;
}

bool MoveFakeQuantize::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
