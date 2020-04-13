// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/pull_transpose_through_fq.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

void ngraph::pass::PullTransposeThroughFQUp::pull_transpose_through_fq() {
    auto data1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto data2 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto data3 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto data4 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto data5 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(data1, data2, data3, data4, data5, 1);
    auto transpose_order = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto transpose = std::make_shared<ngraph::opset1::Transpose>(fq, transpose_order);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto transpose = ngraph::as_type_ptr<ngraph::opset1::Transpose>(m.get_match_root());
        if (!transpose) {
            return false;
        }

        auto const_node = transpose->input(1).get_source_output().get_node_shared_ptr();
        auto const_order = ngraph::as_type_ptr<ngraph::opset1::Constant>(const_node);
        if (!const_order) {
            return false;
        }

        auto fq_node = transpose->input(0).get_source_output().get_node_shared_ptr();
        auto fq = ngraph::as_type_ptr<ngraph::opset1::FakeQuantize>(fq_node);
        if (!fq || fq->output(0).get_target_inputs().size() != 1) {
            return false;
        }

        auto input_shape = fq->input(0).get_source_output().get_shape();

        std::vector<std::shared_ptr<ngraph::Node> > fq_inputs;
        for (size_t i = 0; i < fq->inputs().size(); ++i) {
            std::shared_ptr<ngraph::Node> fq_input;
            fq_input = fq->input(i).get_source_output().get_node_shared_ptr();
            auto fq_input_shape = fq_input->get_shape();
            std::vector<int64_t> unsqueeze_axes;
            for (size_t j = 0; j < input_shape.size() - fq_input_shape.size(); ++j) {
                unsqueeze_axes.push_back(j);
            }
            if (!unsqueeze_axes.empty()) {
                fq_input = std::make_shared<ngraph::opset1::Unsqueeze>(fq_input,
                                                                       opset1::Constant::create(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes));
            }
            fq_input = transpose->copy_with_new_args({fq_input, const_order});
            fq_inputs.push_back(fq_input);
        }

        auto new_fq = fq->copy_with_new_args(fq_inputs);
        new_fq->set_friendly_name(fq->get_friendly_name());
        ngraph::replace_node(transpose, new_fq);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose, "PullTransposeThroughFQUp");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}