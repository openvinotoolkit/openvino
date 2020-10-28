// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <numeric>

#include <legacy/ngraph_ops/fully_connected.hpp>
#include <ngraph/graph_util.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/label.hpp>
#include <ngraph/pattern/op/skip.hpp>
#include <ngraph/util.hpp>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations/utils/utils.hpp>

namespace ngraph {
namespace pass {

class ReshapeFullyConnectedFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ReshapeFullyConnectedFusion : public ngraph::pass::GraphRewrite {
public:
    ReshapeFullyConnectedFusion() : GraphRewrite() {
        construct_reshape_fc();
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> f) override {
        if (!ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(f)) {
            return GraphRewrite::run_on_function(f);
        }
        return false;
    }

private:
    void construct_reshape_fc() {
        auto m_reshape = pattern::wrap_type<opset1::Reshape>(pattern::has_static_shape());
        auto m_fc = pattern::wrap_type<op::FullyConnected>({m_reshape,
                                                            pattern::any_input(),
                                                            pattern::any_input()});

        ngraph::graph_rewrite_callback callback = [=](pattern::Matcher &m) {
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

            auto new_fc = std::make_shared<op::FullyConnected>(reshape->input_value(0),
                                                               fc->input_value(1),
                                                               fc->input_value(2),
                                                               fc->get_shape(),
                                                               fc->output(0).get_element_type());

            new_fc->set_friendly_name(fc->get_friendly_name());
            ngraph::copy_runtime_info({reshape, fc}, new_fc);
            ngraph::replace_node(fc, new_fc);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(m_fc, "ReshapeFullyConnectedFusion");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
