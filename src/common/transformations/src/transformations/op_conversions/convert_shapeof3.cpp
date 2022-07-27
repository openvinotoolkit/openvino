// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_shapeof3.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"

ngraph::pass::ConvertShapeOf3::ConvertShapeOf3() {
    MATCHER_SCOPE(ConvertShapeOf3);
    auto shapeof = pattern::wrap_type<ngraph::opset3::ShapeOf>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto shapeof = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf>(m.get_match_root());
        if (!shapeof) {
            return false;
        }

        Output<Node> last;
        ngraph::NodeVector new_ops;

        auto new_shapeof = std::make_shared<ngraph::opset1::ShapeOf>(shapeof->input_value(0));
        new_ops.push_back(new_shapeof);
        // if the output is the i64 then it matches behavior of the v1::ShapeOf otherwise need to insert Convert
        if (shapeof->get_output_type() == element::i64) {
            last = new_shapeof;
        } else {
            last = std::make_shared<ngraph::opset1::Convert>(new_shapeof, shapeof->get_output_type());
            new_ops.push_back(last.get_node_shared_ptr());
        }

        last.get_node_shared_ptr()->set_friendly_name(shapeof->get_friendly_name());
        ngraph::copy_runtime_info(shapeof, new_ops);
        ngraph::replace_node(shapeof, last.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(shapeof, matcher_name);
    register_matcher(m, callback);
}
