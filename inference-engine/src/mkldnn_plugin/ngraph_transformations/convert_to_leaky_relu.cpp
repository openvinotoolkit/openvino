// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_leaky_relu.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "op/leaky_relu.hpp"

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ConvertToLeakyRelu, "ConvertToLeakyRelu", 0);

MKLDNNPlugin::ConvertToLeakyRelu::ConvertToLeakyRelu() {
    auto prelu = ngraph::pattern::wrap_type<ngraph::opset1::PRelu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                                    ngraph::pattern::any_input(ngraph::pattern::has_static_shape())});

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto prelu = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(m.get_match_root());
        if (!prelu) {
            return false;
        }
        auto slopeNode = std::dynamic_pointer_cast<ngraph::opset1::Constant>(prelu->get_input_node_shared_ptr(1));
        if (slopeNode != nullptr && ngraph::shape_size(prelu->get_input_shape(1)) == 1) {
            const float slope = slopeNode->cast_vector<float>()[0];
            const auto leakyRelu = std::make_shared<MKLDNNPlugin::LeakyReluNode>(prelu->input(0).get_source_output(), slope,
                                                                                 prelu->output(0).get_element_type());
            leakyRelu->set_friendly_name(prelu->get_friendly_name());
            ngraph::copy_runtime_info(prelu, leakyRelu);
            ngraph::replace_node(prelu, leakyRelu);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "ConvertToLeakyRelu");
    this->register_matcher(m, callback);
}
