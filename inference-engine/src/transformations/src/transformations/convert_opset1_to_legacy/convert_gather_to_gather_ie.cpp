// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_gather_to_gather_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::pass::ConvertGatherToGatherIEMatcher::ConvertGatherToGatherIEMatcher() {
    auto gather = ngraph::pattern::wrap_type<opset1::Gather>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto gather = std::dynamic_pointer_cast<ngraph::opset1::Gather>(m.get_match_root());
        if (!gather) {
            return false;
        }

        auto axes_constant = std::dynamic_pointer_cast<ngraph::opset1::Constant>(gather->input_value(2).get_node_shared_ptr());
        if (!axes_constant) {
            return false;
        }
        auto axis = axes_constant->cast_vector<int64_t>()[0];

        // vector of new created nGraph operations
        NodeVector new_ops;

        // if the input with indices is scalar we need to unsqueeze it to 1D so plugins which do not support 0D can
        // execute this layer. Then we need to squeeze the axis dimension to restore original shape of gather output
        auto indices = gather->input_value(1);
        const auto indices_rank = indices.get_partial_shape().rank();
        if (indices_rank.is_dynamic()) {
            return false;
        }

        bool squeeze_gather_output = false;
        if (indices_rank.get_length() == 0) {
            squeeze_gather_output = true;
            indices = std::make_shared<ngraph::opset1::Unsqueeze>(indices, opset1::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(indices.get_node_shared_ptr());
        }

        auto gather_ie = std::make_shared<ngraph::op::GatherIE>(gather->input_value(0), indices, axis);
        new_ops.push_back(gather_ie);

        if (squeeze_gather_output) {
            auto sq = std::make_shared<ngraph::opset1::Squeeze>(gather_ie,
                                                                opset1::Constant::create(element::i64, Shape{1}, {axis}));
            sq->set_friendly_name(gather->get_friendly_name());
            new_ops.push_back(sq);

            ngraph::copy_runtime_info(gather, new_ops);
            ngraph::replace_node(gather, sq);
        } else {
            gather_ie->set_friendly_name(gather->get_friendly_name());
            ngraph::copy_runtime_info(gather, new_ops);
            ngraph::replace_node(gather, gather_ie);
        }
        return true;
    };

    auto m1 = std::make_shared<ngraph::pattern::Matcher>(gather, "ConvertGatherToGatherIE");
    this->register_matcher(m1, callback);
}
