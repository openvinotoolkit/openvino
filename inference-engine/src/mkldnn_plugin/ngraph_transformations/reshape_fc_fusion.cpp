// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_fc_fusion.hpp"
#include "op/fully_connected.hpp"
#include <numeric>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ReshapeFullyConnectedFusion, "ReshapeFullyConnectedFusion", 0);

MKLDNNPlugin::ReshapeFullyConnectedFusion::ReshapeFullyConnectedFusion() {
    auto m_reshape = ngraph::pattern::wrap_type<ngraph::opset1::Reshape>(ngraph::pattern::has_static_shape());
    auto m_fc = ngraph::pattern::wrap_type<MKLDNNPlugin::FullyConnectedNode>({m_reshape, ngraph::pattern::any_input()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        auto & pattern_to_output = m.get_pattern_value_map();
        auto fc = pattern_to_output[m_fc].get_node_shared_ptr();
        auto reshape = pattern_to_output[m_reshape].get_node_shared_ptr();

        // Check that Reshape reshapes 4D tensor to 2D or input shape = output shape
        auto shape_in = reshape->input_value(0).get_shape();
        auto shape_out = reshape->get_shape();
        if (!((shape_in.size() == 4 && reshape->get_shape().size() == 2) || (shape_in == shape_out && !shape_in.empty()))) {
            return false;
        }

        // Check that Weights[O, C*H*W] consistent with Input[N, C, H, W]
        auto shape_w = fc->input_value(1).get_shape();
        if (shape_in[0] != shape_out[0] || std::accumulate(shape_in.begin() + 1, shape_in.end(), size_t{1}, std::multiplies<size_t>()) != shape_w[1]) {
            return false;
        }

        std::shared_ptr<ngraph::Node> new_fc;
        if (fc->get_input_size() == 2) {
            new_fc = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(reshape->input_value(0),
                                                                        fc->input_value(1),
                                                                        fc->get_shape(),
                                                                        fc->output(0).get_element_type());
        } else if (fc->get_input_size() == 3) {
            new_fc = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(reshape->input_value(0),
                                                                        fc->input_value(1),
                                                                        fc->input_value(2),
                                                                        fc->get_shape(),
                                                                        fc->output(0).get_element_type());
        }

        new_fc->set_friendly_name(fc->get_friendly_name());
        ngraph::copy_runtime_info({reshape, fc}, new_fc);
        ngraph::replace_node(fc, new_fc);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_fc, "ReshapeFullyConnectedFusion");
    register_matcher(m, callback);
}
