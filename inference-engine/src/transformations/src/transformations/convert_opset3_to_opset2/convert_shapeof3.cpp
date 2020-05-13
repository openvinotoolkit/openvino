// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset3_to_opset2/convert_shapeof3.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertShapeOf3::convert_shapeof3() {
    auto input = std::make_shared<pattern::op::Label>(element::i64, Shape{1, 1, 1, 1});
    auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto shapeof = std::dynamic_pointer_cast<ngraph::opset3::ShapeOf> (m.get_match_root());
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

    auto m = std::make_shared<ngraph::pattern::Matcher>(shapeof, "ConvertShapeOf3");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}