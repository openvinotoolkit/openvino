// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/convert_gatherND8.hpp"
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

namespace vpu {

ConvertGatherND8ToGatherND5::ConvertGatherND8ToGatherND5() {
    auto gather_nd_v8_pattern = ngraph::pattern::wrap_type<ngraph::opset8::GatherND>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto gather_nd_v8_node = std::dynamic_pointer_cast<ngraph::opset8::GatherND>(m.get_match_root());
        if (!gather_nd_v8_node) {
            return false;
        }
        if (gather_nd_v8_node->get_batch_dims() == 0) {
            auto gather_nd_v5_node = std::make_shared<ngraph::opset5::GatherND>(gather_nd_v8_node->input_value(0),
                                                                gather_nd_v8_node->input_value(1),
                                                                gather_nd_v8_node->get_batch_dims());

            gather_nd_v5_node->set_friendly_name(gather_nd_v8_node->get_friendly_name());
            ngraph::copy_runtime_info(gather_nd_v8_node, gather_nd_v5_node);
            ngraph::replace_node(gather_nd_v8_node, gather_nd_v5_node);
            return true;
        } else {
            return false;
        }
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather_nd_v8_pattern, "ConvertGatherND8ToGatherND5");
    register_matcher(m, callback);
}

}  // namespace vpu
