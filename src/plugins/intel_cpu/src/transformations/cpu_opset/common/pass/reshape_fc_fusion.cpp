// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_fc_fusion.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include <numeric>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

#include "itt.hpp"

ov::intel_cpu::ReshapeFullyConnectedFusion::ReshapeFullyConnectedFusion() {
    MATCHER_SCOPE(ReshapeFullyConnectedFusion);
    auto m_reshape = ngraph::pattern::wrap_type<ngraph::opset1::Reshape>({ngraph::pattern::any_input(ov::pass::pattern::has_static_shape()),
                                                                          ngraph::pattern::any_input()},
                                                                         ngraph::pattern::has_static_shape());
    ngraph::OutputVector fcInputs = {m_reshape, ngraph::pattern::any_input()};
    auto fc = ngraph::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>(fcInputs, ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto fc = std::dynamic_pointer_cast<ov::intel_cpu::FullyConnectedNode>(m.get_match_root());
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
            for (size_t i = 1; i < shape_in.size(); i++)
                newWeightsShape.push_back(shape_in[i]);
        }

        if (newWeightsShape != weightInput.get_shape()) {
            auto newShape = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{newWeightsShape.size()}, newWeightsShape);
            weightInput = std::make_shared<ngraph::opset1::Reshape>(weightInput, newShape, true);
            new_ops.push_back(weightInput.get_node_shared_ptr());
        }

        std::shared_ptr<ngraph::Node> new_fc = std::make_shared<ov::intel_cpu::FullyConnectedNode>(
                                                                        reshape->input_value(0),
                                                                        weightInput,
                                                                        ngraph::Rank(outShape.size()),
                                                                        fc->output(0).get_element_type());
        new_ops.push_back(new_fc);
        new_fc->set_friendly_name(fc->get_friendly_name());
        ngraph::copy_runtime_info({reshape, fc}, new_ops);
        ngraph::replace_node(fc, new_fc);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fc, matcher_name);
    register_matcher(m, callback);
}
