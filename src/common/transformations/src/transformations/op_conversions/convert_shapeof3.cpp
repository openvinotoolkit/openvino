// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_shapeof3.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertShapeOf3::ConvertShapeOf3() {
    MATCHER_SCOPE(ConvertShapeOf3);
    auto shapeof = pattern::wrap_type<ov::op::v3::ShapeOf>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto shapeof = ov::as_type_ptr<ov::op::v3::ShapeOf>(m.get_match_root());
        if (!shapeof) {
            return false;
        }

        Output<Node> last;
        ov::NodeVector new_ops;

        auto new_shapeof = std::make_shared<ov::op::v0::ShapeOf>(shapeof->input_value(0));
        new_ops.push_back(new_shapeof);
        // if the output is the i64 then it matches behavior of the v1::ShapeOf otherwise need to insert Convert
        if (shapeof->get_output_type() == element::i64) {
            last = new_shapeof;
        } else {
            last = std::make_shared<ov::op::v0::Convert>(new_shapeof, shapeof->get_output_type());
            new_ops.push_back(last.get_node_shared_ptr());
        }

        last.get_node_shared_ptr()->set_friendly_name(shapeof->get_friendly_name());
        ov::copy_runtime_info(shapeof, new_ops);
        ov::replace_node(shapeof, last.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(shapeof, matcher_name);
    register_matcher(m, callback);
}
