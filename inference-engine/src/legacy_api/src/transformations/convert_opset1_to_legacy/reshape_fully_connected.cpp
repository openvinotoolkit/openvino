// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/reshape_fully_connected.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "legacy/ngraph_ops/fully_connected.hpp"
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ReshapeFullyConnected, "ReshapeFullyConnected", 0);

ngraph::pass::ReshapeFullyConnected::ReshapeFullyConnected() {
    auto fc = pattern::wrap_type<op::FullyConnected>({pattern::any_input(pattern::has_static_shape()),
                                                      pattern::any_input(),
                                                      pattern::any_input()},
                                                      pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto fc = std::dynamic_pointer_cast<ngraph::op::FullyConnected> (m.get_match_root());
        if (!fc || transformation_callback(fc)) {
            return false;
        }

        auto input_shape = fc->input_value(0).get_shape();
        auto output_shape = fc->get_shape();

        if (input_shape.size() == 2) {
            return false;
        }

        NodeVector new_ops;

        std::vector<int64_t> reshape_shape{-1, static_cast<int64_t>(input_shape.back())};
        auto reshape = std::make_shared<opset1::Reshape>(fc->input_value(0),
                                                         opset1::Constant::create(element::i64, Shape{2}, reshape_shape), true);
        new_ops.push_back(reshape);

        reshape->set_friendly_name(fc->get_friendly_name() + "/Reshape");

        // Calculate output shape for new FullyConnected layer
        // [I, K] * [O, K] = [I, O]
        auto I = reshape->get_shape()[0];
        auto O = fc->input_value(1).get_shape()[0];
        Shape output_shape_new{I, O};

        auto fc_new = std::make_shared<op::FullyConnected>(reshape,
                                                           fc->input_value(1),
                                                           fc->input_value(2),
                                                           output_shape_new,
                                                           fc->get_output_type());
        new_ops.push_back(fc_new);

        if (output_shape != output_shape_new) {
            auto reshape_output = op::util::reshapeTo(fc_new, output_shape);
            new_ops.push_back(reshape_output);
            reshape_output->set_friendly_name(fc->get_friendly_name());
            fc->set_friendly_name(fc->get_friendly_name() + "/FC");
            ngraph::copy_runtime_info(fc, new_ops);
            ngraph::replace_node(fc, reshape_output);
        } else {
            fc_new->set_friendly_name(fc->get_friendly_name());
            ngraph::copy_runtime_info(fc, new_ops);
            ngraph::replace_node(fc, fc_new);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fc, "ReshapeFullyConnected");
    this->register_matcher(m, callback);
}
