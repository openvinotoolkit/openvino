// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/optimize_gather_nd.hpp"

#include <memory>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/op/util/gather_nd_base.hpp>

#include "itt.hpp"

ov::pass::OptimizerGatherND::OptimizerGatherND() {
    MATCHER_SCOPE(OptimizerGatherND);
    auto gather_nd =
        pattern::wrap_type<ov::op::util::GatherNDBase>({pattern::any_input(), pattern::wrap_type<op::v0::Constant>()});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_nd_node = std::dynamic_pointer_cast<ov::op::util::GatherNDBase>(m.get_match_root());
        if (!gather_nd_node)
            return false;

        auto new_indices = op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{}, {0});
        auto new_shape = op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{}, {0});

        auto reshape = make_shared<opset8::Reshape>(gather_nd_node->input_value(0), new_shape, true);

        auto gather =
            make_shared<opset8::Gather>(reshape,
                                        new_indices,
                                        op::v0::Constant::create<int64_t>(element::Type_t::i64, Shape{}, {0}));

        gather->set_friendly_name(gather_nd_node->get_friendly_name());
        ngraph::copy_runtime_info(gather_nd_node, {reshape, gather});
        ngraph::replace_node(gather_nd_node, gather);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather_nd, matcher_name);
    register_matcher(m, callback);
}
