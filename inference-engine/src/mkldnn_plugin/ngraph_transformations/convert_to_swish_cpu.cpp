// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_swish_cpu.hpp"

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "op/swish_cpu.hpp"

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ConvertToSwishCPU, "ConvertToSwishCPU", 0);

MKLDNNPlugin::ConvertToSwishCPU::ConvertToSwishCPU() {
    auto swish = ngraph::pattern::wrap_type<ngraph::opset4::Swish>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto swish = std::dynamic_pointer_cast<ngraph::opset4::Swish> (m.get_match_root());
        if (!swish) {
            return false;
        }
        float beta_value = 1.0;
        if (swish->input_values().size() == 2) {
            auto beta = std::dynamic_pointer_cast<ngraph::opset4::Constant>(swish->get_input_node_shared_ptr(1));

            if (!beta || ngraph::shape_size(swish->get_input_shape(1)) != 1) {
                return false;
            }
            beta_value = beta->cast_vector<float>()[0];
        }

        auto swish_cpu = std::make_shared<MKLDNNPlugin::SwishNode>(swish->input(0).get_source_output(), beta_value);
        swish_cpu->set_friendly_name(swish->get_friendly_name());
        ngraph::copy_runtime_info(swish, swish_cpu);
        ngraph::replace_node(swish, swish_cpu);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(swish, "ConvertToSwishCPU");
    this->register_matcher(m, callback);
}
