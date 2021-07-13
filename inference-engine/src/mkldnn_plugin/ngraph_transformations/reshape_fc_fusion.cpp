// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_fc_fusion.hpp"
#include "op/fully_connected.hpp"
#include <numeric>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ReshapeFullyConnectedFusion, "ReshapeFullyConnectedFusion", 0);

MKLDNNPlugin::ReshapeFullyConnectedFusion::ReshapeFullyConnectedFusion() {
    auto m_reshape = ngraph::pattern::wrap_type<ngraph::opset1::Reshape>(ngraph::pattern::has_static_shape());
    ngraph::OutputVector twoInputs = {m_reshape, ngraph::pattern::any_input()};
    ngraph::OutputVector threeInputs = {m_reshape, ngraph::pattern::any_input(), ngraph::pattern::any_input()};
    auto fcTwoInputs = ngraph::pattern::wrap_type<MKLDNNPlugin::FullyConnectedNode>(twoInputs, ngraph::pattern::has_static_shape());
    auto fcThreeInputs = ngraph::pattern::wrap_type<MKLDNNPlugin::FullyConnectedNode>(threeInputs, ngraph::pattern::has_static_shape());
    const auto fcTwoOrThreeInputs = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{fcTwoInputs, fcThreeInputs});

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto fc = std::dynamic_pointer_cast<MKLDNNPlugin::FullyConnectedNode>(m.get_match_root());
        if (!fc)
            return false;
        auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(fc->get_input_node_shared_ptr(0));
        if (!reshape)
            return false;

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

        ngraph::NodeVector new_ops;
        auto weightInput = fc->input(1).get_source_output();
        ngraph::Shape newWeightsShape;
        const auto outShape = fc->get_shape();
        if (shape_in.size() == 3) {
            newWeightsShape = ngraph::Shape({outShape[2], shape_in[2]});
        } else {
            newWeightsShape.push_back(outShape[1]);
            for (int i = 1; i < shape_in.size(); i++)
                newWeightsShape.push_back(shape_in[i]);
        }

        if (newWeightsShape != weightInput.get_shape()) {
            auto newShape = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{newWeightsShape.size()}, newWeightsShape);
            weightInput = std::make_shared<ngraph::opset1::Reshape>(weightInput, newShape, true);
            new_ops.push_back(weightInput.get_node_shared_ptr());
        }

        std::shared_ptr<ngraph::Node> new_fc;
        if (fc->get_input_size() == 2) {
            new_fc = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(reshape->input_value(0),
                                                                        weightInput,
                                                                        outShape,
                                                                        fc->output(0).get_element_type());
        } else if (fc->get_input_size() == 3) {
            new_fc = std::make_shared<MKLDNNPlugin::FullyConnectedNode>(reshape->input_value(0),
                                                                        weightInput,
                                                                        fc->input_value(2),
                                                                        outShape,
                                                                        fc->output(0).get_element_type());
        } else {
            return false;
        }
        new_ops.push_back(new_fc);
        new_fc->set_friendly_name(fc->get_friendly_name());
        ngraph::copy_runtime_info({reshape, fc}, new_ops);
        ngraph::replace_node(fc, new_fc);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fcTwoOrThreeInputs, "ReshapeFullyConnectedFusion");
    register_matcher(m, callback);
}
