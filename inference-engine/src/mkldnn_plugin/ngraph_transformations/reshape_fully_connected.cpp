// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_fully_connected.hpp"
#include "op/fully_connected.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pattern/op/or.hpp>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ReshapeFullyConnected, "ReshapeFullyConnected", 0);

MKLDNNPlugin::ReshapeFullyConnected::ReshapeFullyConnected() {
    ngraph::OutputVector twoInputs = {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()), ngraph::pattern::any_input()};
    ngraph::OutputVector threeInputs = {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()), ngraph::pattern::any_input(),
                                        ngraph::pattern::any_input()};
    auto fcTwoInputs = ngraph::pattern::wrap_type<MKLDNNPlugin::FullyConnectedNode>(twoInputs, ngraph::pattern::has_static_shape());
    auto fcThreeInputs = ngraph::pattern::wrap_type<MKLDNNPlugin::FullyConnectedNode>(threeInputs, ngraph::pattern::has_static_shape());
    const auto fcTwoOrThreeInputs = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fcTwoInputs, fcThreeInputs});

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto fc = std::dynamic_pointer_cast<MKLDNNPlugin::FullyConnectedNode> (m.get_match_root());
        if (!fc || transformation_callback(fc)) {
            return false;
        }

        auto input_shape = fc->input_value(0).get_shape();
        auto output_shape = fc->get_shape();

        if (input_shape.size() == 2) {
            return false;
        }

        ngraph::NodeVector new_ops;

        std::vector<int64_t> reshape_shape{-1, static_cast<int64_t>(input_shape.back())};
        auto reshape = std::make_shared<ngraph::opset1::Reshape>(fc->input_value(0),
                                                         ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, reshape_shape), true);
        new_ops.push_back(reshape);

        reshape->set_friendly_name(fc->get_friendly_name() + "/Reshape");

        // Calculate output shape for new FullyConnected layer
        // [I, K] * [O, K] = [I, O]
        auto I = reshape->get_shape()[0];
        auto O = fc->input_value(1).get_shape()[0];
        ngraph::Shape output_shape_new{I, O};

        std::shared_ptr<ngraph::Node> fc_new;
        if (fc->get_input_size() == 2) {
            fc_new = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(reshape,
                                                                        fc->input_value(1),
                                                                        output_shape_new,
                                                                        fc->get_output_type());
        } else if (fc->get_input_size() == 3) {
            fc_new = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(reshape,
                                                                        fc->input_value(1),
                                                                        fc->input_value(2),
                                                                        output_shape_new,
                                                                        fc->get_output_type());
        } else {
            return false;
        }
        new_ops.push_back(fc_new);

        if (output_shape != output_shape_new) {
            auto reshape_output = ngraph::op::util::reshapeTo(fc_new, output_shape);
            new_ops.push_back(reshape_output);
            reshape_output->set_friendly_name(fc->get_friendly_name());
            fc_new->set_friendly_name(fc->get_friendly_name() + "/FC");
            ngraph::copy_runtime_info(fc, new_ops);
            ngraph::replace_node(fc, reshape_output);
        } else {
            fc_new->set_friendly_name(fc->get_friendly_name());
            ngraph::copy_runtime_info(fc, new_ops);
            ngraph::replace_node(fc, fc_new);
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fcTwoOrThreeInputs, "ReshapeFullyConnected");
    this->register_matcher(m, callback);
}
